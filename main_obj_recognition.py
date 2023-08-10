import argparse
import os
import random
import shutil
import time
import glob
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

import timm
import wandb

from dataset import ClickMe
from utils import AverageMeter, ProgressMeter, compute_human_alignment
from metrics import accuracy
from configs import DefaultConfigs

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.utils.serialization as xser
except ImportError:
    xm = xmp = pl = xu = None
    
global best_acc
config = DefaultConfigs()

# set running device
if config.tpu == True:
    device = xm.xla_device()
elif torch.cuda.is_available():
    device = 'cuda:{}'.format(config.gpu_id) 
else: 
    device = "cpu"

if config.wandb:
    wandb.login(key="486f67137c1b6905ac11b8caaaf6ecb276bfdf8e")
    wandb.init(
        project="pseudo-clickme",  # set the wandb project where this run will be logged
        
        config={  # track hyperparameters and run metadata
            "learning_rate": config.lr,
            "architecture": config.model_name,
            "dataset": "ImageNet",
            "epochs": config.epochs,
            "mode": config.mode,
        }
    )

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
    assert torch.is_tensor(value), f"the value must be a tensor"
    val = value.item()
    logger.update(val, batch_size)
    
    if config.wandb and var_name != None:
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
        if config.tpu != True:
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
        else:
            xm.add_step_closure(_xla_logging, args=(losses, loss, images.size(0), "training_loss"))
            xm.add_step_closure(_xla_logging, args=(top1, acc1[0], images.size(0), "top1_acc"))
            xm.add_step_closure(_xla_logging, args=(top5, acc5[0], images.size(0), "top5_acc"))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if config.tpu:
            xm.reduce_gradients(optimizer) # Reduce gradients
            # xm.optimizer_step(optimizer, barrier=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (batch_id + 1) % config.interval == 0:
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
            if config.tpu != True:
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
            else:
                xm.add_step_closure(_xla_logging, args=(losses, loss, images.size(0), "val_loss"))
                xm.add_step_closure(_xla_logging, args=(top1, acc1[0], images.size(0), "top1_acc"))
                xm.add_step_closure(_xla_logging, args=(top5, acc5[0], images.size(0), "top5_acc"))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_id + 1) % config.interval == 0:
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
            if config.tpu != True:
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

            if (batch_id + 1) % config.interval == 0:
                progress.display(batch_id + 1)
                
    return top1.avg, losses.avg

def save_checkpoint(state, is_best_acc):
    if not os.path.exists(config.weights):
        os.mkdir(config.best_models) 
    
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models) 
        
    save_dir = config.weights + config.model_name 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = os.path.join(save_dir, "_checkpoint.pth.tar")
    
    if config.tpu == True:
        xser.save(state, filename)
    else: 
        torch.save(state, filename)
        
    if is_best_acc:
        save_dir = config.best_models + config.model_name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        message = os.path.join(save_dir, 'best_acc.pth.tar')
        shutil.copyfile(filename, message)

def main():
    best_acc = 0
        
    torch.set_default_tensor_type('torch.FloatTensor')

    # Model Configurations
    if config.pretrained:
        print("=> using pre-trained model '{}'".format(config.model_name))
        model = timm.create_model(config.model_name, num_classes=1000, pretrained=True)
    else:
        print("=> creating model '{}'".format(config.model_name))
        model = timm.create_model(config.model_name, num_classes=1000, pretrained=False)

    model.to(device)
    if config.tpu:
        broadcast_xla_master_model_param(model, None)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = config.lr,
        momentum = config.momentum,
        weight_decay = config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    if "cuda" in device:
        cudnn.benchmark = True

    # Continue training
    if config.resume:
        if os.path.isfile(config.resume):
            if config.tpu == True:
                checkpoint = xser.load(config.best_models + "model_best.pth.tar")
            else:
                checkpoint = torch.load(config.best_models + "model_best.pth.tar")
            config.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    # Dataset Initialization
    if config.evaluate:
        test_file_paths = glob.glob(os.path.join(config.data_dir, config.test_clickme_paths))
        test_dataset = ClickMe(test_file_paths, is_training=False)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True,
        )
        test(test_loader, model, criterion)
        return
    else:
        if config.mode == "imagenet":
            train_file_paths = glob.glob(os.path.join(config.data_dir, config.train_pseudo_paths))
            val_file_paths = glob.glob(os.path.join(config.data_dir, config.val_pseudo_paths))
        if config.mode == "pseudo":
            train_file_paths = glob.glob(os.path.join(config.data_dir, config.train_pseudo_paths))
            val_file_paths = glob.glob(os.path.join(config.data_dir, config.val_clickme_paths))
        if config.mode == "mix":
            train_file_paths = glob.glob(os.path.join(config.data_dir, config.train_clickme_paths)) 
            train_file_paths += glob.glob(os.path.join(config.data_dir, config.train_pseudo_paths))
            val_file_paths = glob.glob(os.path.join(config.data_dir, config.val_clickme_paths))
            
        train_dataset = ClickMe(train_file_paths, is_training=True)
        val_dataset = ClickMe(val_file_paths, is_training=False)
        
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        print("Global Rank:", global_rank)
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

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
            batch_size = config.batch_size, 
            num_workers = config.num_workers,
            pin_memory = True,
            sampler = train_sampler,
            drop_last = True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size = config.batch_size, 
            num_workers = config.num_workers, 
            pin_memory = True,
            sampler = val_sampler,
            drop_last = True
        )
        
        if config.tpu:
            train_loader = pl.MpDeviceLoader(train_loader, device)
            val_loader = pl.MpDeviceLoader(val_loader, device)
            

    for epoch in range(config.start_epoch, config.epochs):
        print('Epoch: [%d | %d]' % (epoch + 1, config.epochs))

        # para_loader_train = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        # para_loader_val = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
        
        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, model, criterion)

        # Update scheduler
        scheduler.step()

        # save model for best_acc model
        is_best_acc = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            "model_name": config.model_name,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best_acc)

if __name__ == '__main__':
    
    if config.tpu == True:
        tpu_cores_per_node = 8
        xmp.spawn(main(), args=(), nprocs=tpu_cores_per_node)
    else:
        main()
        
    if config.wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks