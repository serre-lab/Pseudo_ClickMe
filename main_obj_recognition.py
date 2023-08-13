import argparse
import os
import random
import shutil
import time
import glob
from itertools import chain
import gc
import warnings
warnings.filterwarnings('ignore')

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

import timm
import wandb

from dataset import ClickMe
from utils import AverageMeter, ProgressMeter, compute_human_alignment
from metrics import accuracy
from configs import Configs as configs

import utils

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.utils.serialization as xser
except ImportError:
    xm = xmp = pl = xu = None
    
best_acc = 0
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

def _xla_logging(logger, value, batch_size, var_name=None):
    val = value.item()
    logger.update(val, batch_size)
    
    if configs.wandb and var_name != None:
        wandb.log({var_name: val})

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
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
        loss = criterion(output, target)

        # measure accuracy and update loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if configs.tpu != True:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
        else:
            xm.add_step_closure(_xla_logging, args=(losses, loss, images.size(0), "training_loss"))
            xm.add_step_closure(_xla_logging, args=(top1, acc1[0], images.size(0), "top1_acc_train"))
            xm.add_step_closure(_xla_logging, args=(top5, acc5[0], images.size(0), "top5_acc_train"))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # if configs.tpu:
        #     xm.reduce_gradients(optimizer) # Reduce gradients
        # optimizer.step()
        if configs.tpu: 
            xm.optimizer_step(optimizer) # # barrier=True is required on single-core training but can be dropped with multiple cores
        else:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.synchronize_between_processes(configs.tpu) # synchronize the tensors across all tpus for every step
        if (batch_id + 1) % configs.interval == 0:
            progress.display(batch_id + 1)
            
    return top1.avg, losses.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_id, (images, heatmaps, target) in enumerate(val_loader):
            images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # compute prediction and loss
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if configs.tpu != True:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
            else:
                xm.add_step_closure(_xla_logging, args=(losses, loss, images.size(0), "val_loss"))
                xm.add_step_closure(_xla_logging, args=(top1, acc1[0], images.size(0), "top1_acc_val"))
                xm.add_step_closure(_xla_logging, args=(top5, acc5[0], images.size(0), "top5_acc_val"))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.synchronize_between_processes(configs.tpu) # synchronize the tensors across all tpus for every step
            if (batch_id + 1) % configs.interval == 0:
                progress.display(batch_id + 1)
                
    return top1.avg, losses.avg

def test(test_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_id, (images, heatmaps, target) in enumerate(test_loader):
            images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # compute prediction and loss
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if configs.tpu != True:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
            else:
                xm.add_step_closure(_xla_logging, args=(losses, loss, images.size(0)))
                xm.add_step_closure(_xla_logging, args=(top1, acc1[0], images.size(0)))
                xm.add_step_closure(_xla_logging, args=(top5, acc5[0], images.size(0)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.synchronize_between_processes(configs.tpu) # synchronize the tensors across all tpus for every step
            if (batch_id + 1) % configs.interval == 0:
                progress.display(batch_id + 1)
                
    return top1.avg, losses.avg

def save_checkpoint(state, is_best_acc):
    '''
    /mnt/disks/bucket/pseudo_clickme/
    |__resnet50
    |    |__imagenet
    |    |    |__ckpt_0.pth
    |    |    |__best.pth
    |    |__mix
    |    |__pseudo
    |...
    '''
    
    def save_model(isXLA, state, filename):
        if isXLA:
            xm.save(state, filename, global_master=True) # save ckpt on master process
        else: 
            torch.save(state, filename)
    
    if not os.path.exists(configs.weights): 
        os.mkdir(configs.weights)  # "/mnt/disks/bucket/pseudo_clickme/"
        
    model_dir = os.path.join(configs.weights, configs.model_name) # "/mnt/disks/bucket/pseudo_clickme/resnet50"
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)
        
    save_dir = os.path.join(model_dir, state['mode']) # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/"
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
        
    filename = os.path.join(save_dir, "ckpt_" + str(state['epoch']) + ".pth.tar") # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/ckpt_#.pth""
    save_model(configs.tpu, state, filename)
    
    if is_best_acc:
        best_filename = os.path.join(save_dir, 'best.pth.tar') # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/best_acc.pth"
        save_model(configs.tpu, state, best_filename)
        
    rmfile = os.path.join(save_dir, "ckpt_" + str(state['epoch'] - configs.ckpt_remain) + ".pth.tar")
    if os.path.exists(rmfile):
        os.remove(rmfile)

def _mp_fn(index):
    global device
    global best_acc
    
    # set running device
    if configs.tpu == True:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = 'cuda:{}'.format(configs.gpu_id) 
    else: 
        device = "cpu"
    
    best_acc = 0
        
    torch.set_default_tensor_type('torch.FloatTensor')

    # Model configsurations
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.model_name))
        model = timm.create_model(configs.model_name, num_classes=1000, pretrained=True)
    else:
        print("=> creating model '{}'".format(configs.model_name))
        model = timm.create_model(configs.model_name, num_classes=1000, pretrained=False)

    model.to(device)
    if configs.tpu:
        broadcast_xla_master_model_param(model, None)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = configs.lr,
        momentum = configs.momentum,
        weight_decay = configs.weight_decay)
    scheduler = StepLR(optimizer, step_size=configs.step_size, gamma=configs.gamma)

    cudnn.benchmark = True

    # Continue training
    if configs.resume:
        if os.path.isfile(configs.resume):
            if configs.tpu == True:
                checkpoint = xm.load(os.path.join(configs.weights, configs.model_name, configs.mode, 'best.pth.tar'))
            else:
                checkpoint = torch.load(os.path.join(configs.weights, configs.model_name, configs.mode, 'best.pth.tar'))
            configs.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Dataset Initialization
    if configs.evaluate:
        test_file_paths = glob.glob(os.path.join(configs.data_dir, configs.test_clickme_paths))
        test_dataset = ClickMe(test_file_paths, is_training=False)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=configs.batch_size, 
            num_workers=configs.num_workers, 
            pin_memory=True,
        )
        test(test_loader, model, criterion)
        return
    
    if configs.mode == "imagenet":
        train_file_paths = glob.glob(os.path.join(configs.data_dir, configs.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(configs.data_dir, configs.val_pseudo_paths))
    if configs.mode == "pseudo":
        train_file_paths = glob.glob(os.path.join(configs.data_dir, configs.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(configs.data_dir, configs.val_clickme_paths))
    if configs.mode == "mix":
        train_file_paths = glob.glob(os.path.join(configs.data_dir, configs.train_clickme_paths)) 
        train_file_paths += glob.glob(os.path.join(configs.data_dir, configs.train_pseudo_paths))
        val_file_paths = glob.glob(os.path.join(configs.data_dir, configs.val_clickme_paths))
        
    train_dataset = ClickMe(train_file_paths, is_training=True)
    val_dataset = ClickMe(val_file_paths, is_training=False)
    
    num_tasks = utils.get_world_size(configs.tpu)
    global_rank = utils.get_rank(configs.tpu)

    print("Global Rank:", global_rank)
    sampler_rank = global_rank

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
        batch_size = configs.batch_size, 
        num_workers = configs.num_workers,
        pin_memory = True,
        sampler = train_sampler,
        drop_last = True # DataParallel cores must run the same number of batches each, and only full batches are allowed.
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = configs.batch_size, 
        num_workers = configs.num_workers, 
        pin_memory = True,
        sampler = val_sampler,
        drop_last = True
    )
    
    if configs.tpu:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    for epoch in range(configs.start_epoch, configs.epochs):
        print('Epoch: [%d | %d]' % (epoch + 1, configs.epochs))

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, model, criterion)

        # Update scheduler
        scheduler.step()

        # save model for best_acc model
        if epoch < configs.epochs // 2: continue
        is_best_acc = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            "model_name": configs.model_name,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'mode':configs.mode
        }, is_best_acc)
        
        gc.collect()

if __name__ == '__main__':
    if configs.wandb:
        wandb.login(key="486f67137c1b6905ac11b8caaaf6ecb276bfdf8e")
        wandb.init(
            project="pseudo-clickme",  # set the wandb project where this run will be logged
            
            config={  # track hyperparameters and run metadata
                "learning_rate": configs.lr,
                "architecture": configs.model_name,
                "dataset": "ImageNet",
                "epochs": configs.epochs,
                "mode": configs.mode,
            }
        )
    
    if configs.tpu == True:
        tpu_cores_per_node = 1
        xmp.spawn(_mp_fn, args=(), nprocs=tpu_cores_per_node) # cannot call xm.xla_device() before spawing
    else:
        _mp_fn(0)
        
    if configs.wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks
        
    print("***** DONE! *****")