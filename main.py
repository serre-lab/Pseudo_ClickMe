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

import timm
import wandb

from dataset import ClickMe
from utils import AverageMeter, ProgressMeter, compute_human_alignment
from metrics import accuracy
from configs import DefaultConfigs
from loss import harmonizer_loss, harmonization_eval

best_acc = 0
best_human_alignment = 0
config = DefaultConfigs()
device = 'cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else "cpu"

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
        
        # compute losses
        harmonization_loss, cce_loss = harmonizer_loss(model, images, target, heatmaps, criterion)

        # measure accuracy and update loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(cce_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # update
        optimizer.zero_grad()
        harmonization_loss.backward()
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
    alignment = AverageMeter('Alignment', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, alignment],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_id, (images, heatmaps, target) in enumerate(val_loader):
        images, heatmaps, target = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), target.to(device, non_blocking=True)

        '''
        # compute prediction and loss
        images.requires_grad = True
        output = model(images)

        # measure accuracy and record loss
        cce_loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(cce_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # compute saliency maps and measure human alignment
        correct_class_scores = output.gather(1, target.view(-1, 1)).squeeze()
        saliency_loss = torch.sum(correct_class_scores)
        saliency_loss.backward(retain_graph=True) # compute the gradients while retain the graph
        grads = torch.abs(images.grad)
        saliency_maps, _ = torch.max(grads, dim=1, keepdim=True) # saliency map (N, C, H, W) -> (N, 1, H, W)
        human_alignment = compute_human_alignment(saliency_maps, heatmaps)
        alignment.update(human_alignment, images.size(0))
        images.grad.zero_() # reset the gradients
        '''
        
        output, cce_loss, alignment_score = harmonization_eval(model, images, target, heatmaps, criterion)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(cce_loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        alignment.update(alignment_score, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id + 1) % config.interval == 0:
            progress.display(batch_id + 1)

    return top1.avg, losses.avg, alignment.avg

def save_checkpoint(state, is_best_acc, is_best_alignment):
    save_dir = config.weights + config.model_name 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = os.path.join(save_dir, "_checkpoint.pth.tar")
    
    torch.save(state, filename)
    if is_best_acc:
        save_dir = config.best_models + config.model_name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        message = os.path.join(save_dir, 'best_acc.pth.tar')
        shutil.copyfile(filename, message)
        
    if is_best_alignment:
        save_dir = config.best_models + config.model_name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        message = os.path.join(save_dir, 'best_alignment.pth.tar')
        shutil.copyfile(filename, message)

def main():
    global best_acc
    global best_human_alignment

    # Model Configurations
    if config.pretrained:
        print("=> using pre-trained model '{}'".format(config.model_name))
        # model = models.__dict__[config.model_name](pretrained=True)
        model = timm.create_model(config.model_name, num_classes=1000, pretrained=True)
    else:
        print("=> creating model '{}'".format(config.model_name))
        # model = models.__dict__[config.model_name]()
        model = timm.create_model(config.model_name, num_classes=1000, pretrained=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr = config.lr,
    #     momentum = config.momentum,
    #     weight_decay = config.weight_decay)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = config.lr,
        weight_decay = config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    cudnn.benchmark = True

    # Continue training
    if config.resume:
        if os.path.isfile(config.resume):
            checkpoint = torch.load(config.best_models + "model_best.pth.tar")
            config.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_human_alignment = checkpoint['best_human_alignment']
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
        test_dataset = ClickMe(test_file_paths)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True,
        )
        validate(val_loader, model, criterion)
        return
    else:
        assert config.mode in ["imagenet", "pseudo", "mix"], f"Please check the data paths!"
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

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True,
            shuffle=False
        )

    for epoch in range(config.start_epoch, config.epochs):
        print('Epoch: [%d | %d]' % (epoch + 1, config.epochs))

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_acc, val_loss, human_alignment = validate(val_loader, model, criterion)

        # Update scheduler
        scheduler.step()

        # save model for best_acc model
        is_best_acc = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        is_best_alignment = human_alignment > best_human_alignment
        best_human_alignment = max(human_alignment, best_human_alignment)
        save_checkpoint({
            'epoch': epoch + 1,
            "model_name": config.model_name,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'alignment': human_alignment,
            'best_human_alignment': best_human_alignment,
            'optimizer': optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best_acc, is_best_alignment)

if __name__ == '__main__':
    main()