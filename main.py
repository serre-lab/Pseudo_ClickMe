import os
import gc
import random
import shutil
import time
import glob
import pathlib
import argparse
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import timm
import wandb

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import utils
from dataset import ClickMe
from metrics import accuracy
from loss import harmonizer_loss, harmonization_eval
from utils import AverageMeter, ProgressMeter, compute_human_alignment

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.utils.serialization as xser
except ImportError:
    xm = xmp = pl = xu = None
    
best_acc, best_human_alignment = 0, 0 
device = None

def broadcast_xla_master_model_param(model, args):
    """
    Broadcast the model parameters from master process to other processes;
    Set all params in non-master devices to zero so that all_reduce is
    equivalent to broadcasting parameters from master to other devices.
    """
    parameters_and_buffers = []
    is_master = xm.is_master_ordinal(local=False)
    for p in chain(model.parameters(), model.buffers()):
        scale = 1 if is_master else 0
        scale = torch.tensor(scale, dtype=p.data.dtype, device=p.data.device)
        p.data.mul_(scale)
        parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()
    xm.rendezvous("broadcast_xla_master_model_param")

def _xla_logging(loggers, values, batch_size, args, global_rank, var_names=None):
    pairs = {}
    if not var_names: var_names = [None] * len(values)
    for logger, value, var_name in zip(loggers, values, var_names):
        val = value.item()
        logger.update(val, batch_size)
        if var_name: pairs[var_name] = val
        
    if global_rank == 0 and args.wandb and var_name != None: # just update values on the main process
        for var_name, value in zip(var_names, values):
            wandb.log(pairs)

def train(train_loader, model, criterion, optimizer, epoch, args, global_rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    cce_losses = AverageMeter('CCE_Loss', ':.2e')
    hmn_losses = AverageMeter('HMN_Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, cce_losses, hmn_losses, top1],
        prefix="Train: [{}]".format(epoch+1))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_id, (images, heatmaps, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # compute prediction output
        output = model(images)
        
        # compute losses
        harmonization_loss, cce_loss = harmonizer_loss(model, images, target, heatmaps, criterion)

        # measure accuracy and update loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        if args.tpu != True:
            hmn_losses.update(harmonization_loss.item(), images.size(0))
            cce_losses.update(cce_loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
        else:
            if args.epochs <= args.logger_update or (batch_id + 1) % args.logger_update == 0: # otherwise, passing values from TPU to CPU will be very slow
                var_names = ["harmonization_loss", "train_cce_loss", "top1_acc"]
                loggers = [hmn_losses, cce_losses, top1]
                values = [harmonization_loss, cce_loss, acc1[0]]
                xm.add_step_closure(_xla_logging, args=(loggers, values, images.size(0), args, global_rank, var_names))
                
        # update
        optimizer.zero_grad()
        harmonization_loss.backward()
        if args.tpu: 
            xm.optimizer_step(optimizer) # # barrier=True is required on single-core training but can be dropped with multiple cores
        else:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id + 1) % args.interval == 0:
            progress.synchronize_between_processes(args.tpu) # synchronize the tensors across all tpus for every step
            progress.display(batch_id + 1, args.tpu)

    return top1.avg, hmn_losses.avg

def validate(val_loader, model, criterion, args, global_rank):
    batch_time = AverageMeter('Time', ':6.3f')
    cce_losses = AverageMeter('CCE-Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    alignment = AverageMeter('Alignment', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, cce_losses, top1, alignment],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_id, (images, heatmaps, target) in enumerate(val_loader):
        images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # evaluation
        output, cce_loss, alignment_score = harmonization_eval(model, images, target, heatmaps, criterion)
        acc1 = accuracy(output, target, topk=(1, ))[0]
        
        if args.tpu != True:
            cce_losses.update(cce_loss.item(), images.size(0))
            alignment.update(alignment_score.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
        else:
            var_names = ["val_cce_loss", "alignment", "top1_acc"]
            loggers = [cce_losses, alignment, top1]
            values = [cce_loss, alignment_score, acc1[0]]
            xm.add_step_closure(_xla_logging, args=(loggers, values, images.size(0), args, global_rank, var_names))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id + 1) % (args.interval // 5) == 0:
            progress.synchronize_between_processes(args.tpu) # synchronize the tensors across all tpus for every step
            progress.display(batch_id + 1, args.tpu)

    return top1.avg, losses.avg, alignment.avg

def test(test_loader, model, criterion, args, global_rank):
    batch_time = AverageMeter('Time', ':6.3f')
    cce_losses = AverageMeter('CCE-Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    alignment = AverageMeter('Alignment', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, cce_losses, top1, alignment],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_id, (images, heatmaps, target) in enumerate(test_loader):
        images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        output, cce_loss, alignment_score = harmonization_eval(model, images, target, heatmaps, criterion)
        acc1 = accuracy(output, target, topk=(1, ))[0]
        
        if args.tpu != True:
            cce_losses.update(cce_loss.item(), images.size(0))
            alignment.update(alignment_score.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
        else:
            loggers = [cce_losses, alignment, top1]
            values = [cce_loss, alignment_score, acc1[0]]
            xm.add_step_closure(_xla_logging, args=(loggers, values, images.size(0), args, global_rank))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.synchronize_between_processes(args.tpu) # synchronize the tensors across all tpus for every step
        progress.display(batch_id + 1, args.tpu)

    return top1.avg, losses.avg, alignment.avg

def _mp_fn(index, args):
    global device
    global best_acc
    global best_human_alignment
    
    best_acc = 0
    best_human_alignment = 0
    global_rank = utils.get_rank(args.tpu)
    
    # enable wandb
    if args.wandb and global_rank == 0:
        wandb.login(key="486f67137c1b6905ac11b8caaaf6ecb276bfdf8e")
        wandb.init(
            project="pseudo-clickme",  # set the wandb project where this run will be logged
            entity="serrelab",
            config={  # track hyperparameters and run metadata
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "dataset": "ImageNet",
                "epochs": args.epochs,
                "mode": args.mode,
                "pretrained": args.pretrained
            }
        )
    
    # set running device
    if args.tpu == True:
        device = xm.xla_device() # device on current procoess
    elif torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu_id) 
    else: 
        device = "cpu"
    
    torch.set_default_tensor_type('torch.FloatTensor')
    
    seed = args.seed + utils.get_rank(args.tpu)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model configsurations
    if args.pretrained:
        if args.tpu:
            xm.master_print("=> using pre-trained model '{}'".format(args.model_name))
        else:
            print("=> using pre-trained model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=True)
    else:
        if args.tpu:
            xm.master_print("=> creating model '{}'".format(args.model_name))
        else:
            print("=> creating model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=False)

    model.to(device)
    if args.tpu:
        broadcast_xla_master_model_param(model, None)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = args.learning_rate,
        weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    cudnn.benchmark = True
       
    # Continue training
    if args.resume:
        if os.path.isfile(args.resume):
            if args.tpu == True:
                checkpoint = xm.load(os.path.join(args.weights, args.model_name, args.mode, 'best.pth.tar'))
            else:
                checkpoint = torch.load(os.path.join(args.weights, args.model_name, args.mode, 'best.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_human_alignment = checkpoint['best_human_alignment']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

     # Dataset Initialization
    if args.evaluate:
        test_file_paths = glob.glob(os.path.join(args.data_dir, args.test_clickme_paths))
        test_dataset = ClickMe(test_file_paths, is_training=False)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            pin_memory=True,
        )
        test(test_loader, model, criterion, args)
        return
    
    assert args.mode in ["imagenet", "pseudo", "mix"], f"Please check the data paths!"
    if args.mode == "imagenet":
        train_file_paths = glob.glob(os.path.join(args.data_dir, args.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(args.data_dir, args.val_pseudo_paths))
    if args.mode == "pseudo":
        train_file_paths = glob.glob(os.path.join(args.data_dir, args.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(args.data_dir, args.val_clickme_paths))
    if args.mode == "mix":
        train_file_paths = glob.glob(os.path.join(args.data_dir, args.train_clickme_paths)) 
        train_file_paths += glob.glob(os.path.join(args.data_dir, args.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(args.data_dir, args.val_clickme_paths))
                
    train_dataset = ClickMe(train_file_paths, is_training=True)
    val_dataset = ClickMe(val_file_paths, is_training=False)
    
    print("Global Rank:", global_rank)
    sampler_rank = global_rank
    
    num_tasks = utils.get_world_size(args.tpu)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas = num_tasks,
        rank = sampler_rank, 
        shuffle = True)

    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas = num_tasks,
        rank = sampler_rank, 
        shuffle = False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers,
        pin_memory = False,
        sampler = train_sampler,
        drop_last = True # DataParallel cores must run the same number of batches each, and only full batches are allowed.
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
        pin_memory = False,
        sampler = val_sampler,
        drop_last = True
    )
    
    if args.tpu:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    for epoch in range(args.start_epoch, args.epochs):
        if args.tpu:
            xm.master_print('Epoch: [%d | %d]' % (epoch + 1, args.epochs))
        else:
            print('Epoch: [%d | %d]' % (epoch + 1, args.epochs))
            
        epoch_s = time.time()

        # train for one epoch
        if args.tpu:
            xm.master_print('Training Starts...')
        else:
            print('Training Starts...')
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args, global_rank)

        # evaluate on validation set
        if args.tpu:
            xm.master_print('Val Starts...')
        else:
            print('Val Starts...')
        val_acc, val_loss, human_alignment = validate(val_loader, model, criterion, args, global_rank)

        # Update scheduler
        scheduler.step()
        if args.tpu:
            xm.master_print('Scheduler Updates')
        else:
            print('Scheduler Updates')
        
        epoch_e = time.time()
        
        if args.tpu:
            xm.master_print("******************* Save CKPT Started *******************")

        # save model for best_acc model
        # if epoch < args.epochs // 2: continue
        is_best_acc = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        is_best_alignment = human_alignment > best_human_alignment
        best_human_alignment = max(human_alignment, best_human_alignment)

        '''
        # No need to check whether it is on the main process. Torch_XLA automatically does that for us
        # 'rendezvous' in xm.save() makes sure all processes are at the same stage
        # if process A is at here, it will wait for other processes until here. 
        '''
        utils.save_checkpoint({
            'epoch': epoch + 1,
            "model_name": args.model_name,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'alignment': human_alignment,
            'best_human_alignment': best_human_alignment,
            'optimizer': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'mode':args.mode
        }, is_best_acc, is_best_alignment, epoch+1, global_rank, args)
        
        if args.tpu: 
            xm.master_print("******************* Save CKPT Finished *******************")
        
        if args.tpu:
            xm.master_print("Epoch {}: {} seconds".format(str(epoch+1), str(epoch_e - epoch_s)))
        else:
            print("Epoch {}: {} seconds".format(str(epoch+1), str(epoch_e - epoch_s)))

if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser('Harmonization PyTorch Scripts for TPU Training', add_help=False)
    parser.add_argument("-dd", "--data_dir", required = False, type = str, 
                        default = '/mnt/disks/clickme-with-pseudo/test',
                        # choices=[
                        #     '/mnt/disks/clickme-with-pseudo/test', # small-scale data examples
                        #     '/mnt/disks/clickme-with-pseudo/'
                        # ],
                        help="please enter a data directory")
    parser.add_argument("--train_pseudo_paths", required = False, type = str, 
                        default = 'PseudoClickMe/train/*.pth', help = "please enter a data directory")
    parser.add_argument("--train_clickme_paths", required = False, type = str, 
                        default = 'ClickMe/train/*.pth', help = "please enter a data directory")
    parser.add_argument("--val_pseudo_paths", required = False, type = str, 
                        default = 'PseudoClickMe/val/*.pth', help = "please enter a data directory")
    parser.add_argument("--val_clickme_paths", required = False, type = str, 
                        default = 'ClickMe/val/*.pth', help = "please enter a data directory")
    parser.add_argument("--test_clickme_paths", required = False, type = str, 
                        default = 'ClickMe/test/*.pth', help = "please enter a data directory")
    parser.add_argument("-wt", "--weights", required = False, type = str,
                        default = '/mnt/disks/bucket/pseudo_clickme/',
                        help = "please enter a directory save checkpoints")
    parser.add_argument("-mn", "--model_name", required = False, type = str,
                        default = 'resnet50',
                        help="Please specify a model architecture according to TIMM")
    parser.add_argument("-md", "--mode", required=False, type = str,
                        default = 'imagenet', 
                        choices = [
                            "pseudo",  # pure imagenet images with pseudo clickme maps
                            "mix",     # imagenet images with pseudo and real clickme maps
                            "imagenet" # imagenet images 
                        ],
                        help="'pseudo', 'mix' or 'imagenet'?")
    parser.add_argument("-ep", "--epochs", required=False, type = int, 
                        default = 3,
                        help="Number of Epochs")
    parser.add_argument("-sp", "--start_epoch", required=False, type = int, 
                        default = 0,
                        help="start epoch is usually used with 'resume'")
    parser.add_argument("-bs", "--batch_size", required=False, type = int,
                        default = 4,
                        help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", required=False, type = float,
                        default = 0.1,
                        help="Learning Rate")
    parser.add_argument("-mt", "--momentum", required=False, type = float,
                        default = 0.9,
                        help="SGD momentum")
    parser.add_argument("-ss", "--step_size", required=False, type = int,
                        default = 25,
                        help="learning rate scheduler")
    parser.add_argument("-gm", "--gamma", required=False, type = float,
                        default = 0.1,
                        help="scheduler parameters, which decides the change of learning rate ")
    parser.add_argument("-wd", "--weight_decay", required=False, type = int,
                        default = 1e-5,
                        help="weight decay, regularization")
    parser.add_argument("-iv", "--interval", required=False, type = int,
                        default = 2,
                        help="Step interval for printing logs")
    parser.add_argument("-nw", "--num_workers", required=False, type = int,
                        default = 8,
                        help="number of workers in dataloader")
    parser.add_argument("-gid", "--gpu_id", required=False, type = int,
                        default = 1,
                        help="specify gpu id for single gpu training")
    parser.add_argument("-tc", "--tpu_cores_per_node", required=False, type = int,
                        default = 1,
                        help="specify the number of tpu cores")
    parser.add_argument("-ckpt", "--ckpt_remain", required=False, type = int,
                        default = 5,
                        help="how many checkpoints can be saved at most?")
    parser.add_argument("-lu", "--logger_update", required=False, type = int,
                        default = 10,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-sd", "--seed", required=False, type = int,
                        default = 42,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-ev", "--evaluate", required=False, type = str,
                        default = False,
                        action = utils.str2bool,
                        help="Whether to evaluate a model")
    parser.add_argument("-pt", "--pretrained", required=False, type = str,
                        default = False,
                        action = utils.str2bool,
                        help="Whether to use pretrained model from TIMM")
    parser.add_argument("-rs", "--resume", required=False, type = str,
                        default = False,
                        action = utils.str2bool,
                        help="Whether to continue (usually used with 'evaluate')")
    parser.add_argument("-gt", "--tpu", required=False, type = str,
                        default = False,
                        action = utils.str2bool,
                        help="Whether to use Google Cloud Tensor Processing Units")
    parser.add_argument("-wb", "--wandb", required=False, type = str,
                        default = False,
                        action = utils.str2bool,
                        help="Whether to W&B to record progress")
    
    # modify the configurations according to args parser
    args = parser.parse_args()
        
    start_time = time.time()
    
    # start running
    if args.tpu == True:
        tpu_cores_per_node = args.tpu_cores_per_node  # a TPUv3 device contains 4 chips and 8 cores in total
        xmp.spawn(_mp_fn, args=(args,), nprocs=tpu_cores_per_node) # cannot call xm.xla_device() before spawing
                                                                   # don't forget comma args=(args,)
    else:
        _mp_fn(0, args)
        
    if args.wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks
        
    end_time = time.time()
    print('Total hours: ', round((end_time - start_time) / 3600, 1))
    print("****************************** DONE! ******************************")