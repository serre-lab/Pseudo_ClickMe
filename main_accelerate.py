'''
Terminal cmd:
- init
    accelerate config
- Train
    accelerate launch --main_process_port 29501 main_accelerate.py -dd '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full' -mn 'resnet50' -md "clickme" -ep 100 -bs 256 -pt True -wb True
    accelerate launch --main_process_port 29501 main_accelerate.py -mn 'resnet50' -md  "clickme" -ep 10 -bs 8 -pt True
    accelerate launch --main_process_port 29501 main_accelerate.py -mn 'resnet50' -md  "clickme" -ep 1 -bs 8 -pt True
- Eval
    accelerate launch main_accelerate.py -dd '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full' -mn 'resnet50' -md "clickme" -bs 256 -ev True -rs True
    accelerate launch main_accelerate.py -dd '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full' -mn 'resnet50' -md "clickme" -bs 64 -ev True -rs True -pt True
    accelerate launch --main_process_port 29501 main_accelerate.py -dd '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full' -mn 'resnet50' -md "clickme" -bs 64 -ev True -rs True -bt "best_acc"
    accelerate launch --main_process_port 29501 main_accelerate.py -dd '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full' -mn 'resnet50' -md "clickme" -bs 64 -ev True -rs True -bt "best_alignment"
    accelerate launch --main_process_port 29501 main_accelerate.py -mn 'resnet50' -md  "clickme" -bs 4 -pt True -ev True -rs True
    accelerate launch --main_process_port 29501 main_accelerate.py -mn 'resnet50' -md  "clickme" -bs 4 -pt True -ev True
'''

import os
import gc
import random
import time
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import timm
import wandb
from accelerate import Accelerator
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader

import utils
from dataset import ClickMe
from metrics import accuracy
from loss import harmonizer_loss, harmonization_eval
from utils import AverageMeterAcc, ProgressMeterAcc, compute_human_alignment, save_checkpoint_accelerator, CosineAnnealingWithWarmup, EarlyStopping

def accelerate_logging(loggers, values, batch_size, accelerator, args, var_names=None, status="train", isLastBatch=False):
    pairs = {}
    if not var_names: var_names = [None] * len(values)
    for logger, value, var_name in zip(loggers, values, var_names):
        val = value.item()
        logger.update(val, batch_size)
        if var_name: pairs[var_name] = val

    if status == "train" or (status == "val" and isLastBatch):
        if accelerator.is_main_process and args.wandb and var_names: # just update values on the main process
            for var_name, value in zip(var_names, values):
                wandb.log(pairs)
    
def train(train_loader, model, criterion, optimizer, lr_scheduler, accelerator, args):
    train_cce_losses = AverageMeterAcc('CCE_Loss', ':.2f')
    train_hmn_losses = AverageMeterAcc('HMN_Loss', ':.2f')
    train_top1 = AverageMeterAcc('Acc@1', ':6.3f')
    train_progress = ProgressMeterAcc(
        len(train_loader),
        [train_cce_losses, train_hmn_losses, train_top1],
        prefix="Train: ")
    
    # switch to train mode
    model.train()
    
    device = accelerator.device
    
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc="Train", position=0, leave=True)
    else:
        pbar = train_loader

    pyramid_grad_norm, cce_grad_norm, hmn_grad_norm = None, None, None
    for batch_id, (images, heatmaps, targets) in enumerate(pbar):
        images, heatmaps, targets = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # compute losses
        hmn_loss, pyramid_loss, cce_loss, outputs = harmonizer_loss(model, images, targets, heatmaps, criterion, accelerator=accelerator)
        # accelerator.print(format(images_grad_norm_hmn, '.3e'), format(images_grad_norm_cce, '.3e'), format(images_grad_norm_pyramid, '.3e'))

        # # log gradient norm
        # optimizer.zero_grad()
        # accelerator.backward(pyramid_loss, retain_graph=True)
        # pyramid_grad_norm = utils.compute_gradient_norm(model)
        
        # optimizer.zero_grad()
        # accelerator.backward(cce_loss, retain_graph=True)
        # cce_grad_norm = utils.compute_gradient_norm(model)
        
        # Update 
        optimizer.zero_grad()
        accelerator.backward(hmn_loss)
        
        hmn_grad_norm = utils.compute_gradient_norm(model)
  
        optimizer.step()
        lr_scheduler.step()

        # accelerator.print(pyramid_loss, cce_loss, format(pyramid_grad_norm, '.6f'), format(cce_grad_norm, '.6f'), format(hmn_grad_norm, '.6f'))

        # log gradient norm
        if accelerator.is_main_process and args.wandb: # just update values on the main process
            wandb.log({
                "pyramid_grad_norm": pyramid_grad_norm if pyramid_grad_norm else 0,
                "cce_grad_norm": cce_grad_norm if cce_grad_norm else 0,
                "hmn_grad_norm": hmn_grad_norm if hmn_grad_norm else 0,
                # "images_grad_norm_hmn_loss": images_grad_norm_hmn, 
                # "images_grad_norm_cce_loss": images_grad_norm_cce, 
                # "images_grad_norm_pyramid_loss": images_grad_norm_pyramid
            })
        
        # Log
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        var_names = ["train_hmn_loss", "train_cce_loss", "train_top1_acc"] if not args.evaluate else None
        loggers = [train_hmn_losses, train_cce_losses, train_top1]
        values = [hmn_loss, cce_loss, acc1[0]]
        accelerate_logging(loggers, values, images.size(0), accelerator, args, var_names=var_names)
        
        # Log learning rate
        if accelerator.is_main_process and args.wandb: # just update values on the main process
            wandb.log({"lr": lr_scheduler.get_last_lr()[0]})

        # print(f"[Train] Acc@1: {acc1[0]:.4f} %, Harmonization Loss: {hmn_loss:.4f}, CCE Loss: {cce_loss:.4f}")

        # Synchronize results
        accelerator.wait_for_everyone()
        assert accelerator.sync_gradients
        train_progress.synchronize_between_processes(accelerator) # synchronize the tensors across all tpus for every step
        
    # progress.display(len(train_loader), accelerator)
    avg_top1, avg_hmn_loss, avg_cce_loss = train_top1.avg, train_hmn_losses.avg, train_cce_losses.avg
    del train_cce_losses, train_hmn_losses, train_top1, train_progress
    return avg_top1, avg_hmn_loss, avg_cce_loss

def evaluate(eval_loader, model, criterion, accelerator, args):

    eval_cce_losses = AverageMeterAcc('CCE_Loss', ':.2f')
    eval_top1 = AverageMeterAcc('Acc@1', ':6.2f')
    eval_alignment = AverageMeterAcc('Alignment', ':6.3f')
    eval_progress = ProgressMeterAcc(
        len(eval_loader),
        [eval_cce_losses, eval_top1, eval_alignment],
        prefix='Eval:  ')
    
    # switch to evaluate mode
    model.eval()
    
    device = accelerator.device
    status = "test" if args.evaluate else "val"
    
    if accelerator.is_main_process:
        pbar = tqdm(eval_loader, desc="Eval ", position=0, leave=True)
    else:
        pbar = eval_loader

    for batch_id, (images, heatmaps, targets) in enumerate(pbar):
        images, heatmaps, targets = images.to(device, non_blocking=True), heatmaps.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        outputs, cce_loss, alignment_score = harmonization_eval(model, images, targets, heatmaps, criterion, accelerator=accelerator)
        
        # Log
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        var_names = ["eval_cce_loss", "eval_alignment", "eval_top1_acc"]
        loggers = [eval_cce_losses, eval_alignment, eval_top1]
        values = [cce_loss, alignment_score, acc1[0]]
        accelerate_logging(loggers, values, images.size(0), accelerator, args, var_names=var_names, status=status, isLastBatch=batch_id==(len(eval_loader)-1))
        
        # Synchronize results
        accelerator.wait_for_everyone()
        eval_progress.synchronize_between_processes(accelerator) # synchronize the tensors across all tpus for every step
        
    # progress.display(len(eval_loader), accelerator)
    avg_top1, avg_cce_loss, avg_alignment = eval_top1.avg, eval_cce_losses.avg, eval_alignment.avg
    del eval_cce_losses, eval_top1, eval_alignment, eval_progress
    return avg_top1, avg_cce_loss, avg_alignment

def run(args):
    start_time = time.time()
    
    global best_acc
    global best_human_alignment
    
    best_acc = 0
    best_human_alignment = 0
    
    # Initialize accelerator
    accelerator = Accelerator(cpu=False, mixed_precision='no') # choose from 'no', 'fp8', 'fp16', 'bf16'
    
    # enable wandb
    if args.wandb and accelerator.is_main_process and not args.evaluate:
        wandb.login(key="486f67137c1b6905ac11b8caaaf6ecb276bfdf8e")
        wandb.init(
            project="pseudo-clickme",  # set the wandb project where this run will be logged
            entity="serrelab",
            config={                   # track hyperparameters and run metadata
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "dataset": "ClickMe",
                "epochs": args.epochs,
                "mode": args.mode,
                "pretrained": args.pretrained
            }
        )
    
    # Set the random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Choose whether to use pretrained weights
    if args.pretrained:
        accelerator.print("=> using pre-trained model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=args.pretrained)
    else:
        accelerator.print("=> creating model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=args.pretrained)
        
        # Continue training
        if args.resume or args.evaluate:
            ckpt_path = os.path.join(args.weights, args.model_name, args.mode, args.best_model + '.pth.tar')
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['state_dict'])
                accelerator.print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
            else:
                accelerator.print("=> no checkpoint found at '{}'".format(args.resume))
                return
            
     # Initialization
    if args.evaluate:
        test_file_paths = glob.glob(os.path.join(args.data_dir, args.test_clickme_paths)) + glob.glob(os.path.join(args.data_dir, args.val_clickme_paths))
        # test_file_paths = glob.glob(os.path.join(args.data_dir, args.test_clickme_paths)) 
        test_dataset = ClickMe(test_file_paths, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        model, criterion, test_loader = accelerator.prepare(model, criterion, test_loader)
    else:
        assert args.mode in ["imagenet", "pseudo", "mix", "clickme"], f"Please check the data paths!"
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
        if args.mode == "clickme":
            train_file_paths = glob.glob(os.path.join(args.data_dir, args.train_clickme_paths)) 
            val_file_paths = glob.glob(os.path.join(args.data_dir, args.val_clickme_paths))
            
        train_dataset = ClickMe(train_file_paths, is_training=True)
        val_dataset = ClickMe(val_file_paths, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = False, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = False, drop_last = True)

        # Instantiate optimizer and learning rate scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=False)
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=False)
        steps_per_epoch = len(train_loader) 
        # lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs*steps_per_epoch, eta_min=1e-6)
        # lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs*steps_per_epoch, eta_min=0) 
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs*steps_per_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer = optimizer, min_lrs = [1e-6], first_cycle_steps = args.epochs*steps_per_epoch, warmup_steps = args.warmup*steps_per_epoch, gamma = 0.9)
        
        model, optimizer, lr_scheduler, criterion, train_loader, val_loader = accelerator.prepare(
            model, optimizer, lr_scheduler, criterion, train_loader, val_loader)
    
    if args.evaluate:
        model.eval()
        test_acc, test_loss, alignment_score = evaluate(test_loader, model, criterion, accelerator, args)
        accelerator.print(f"[Test: ] Accuracy: {test_acc:.2f} %; alignment score is: {alignment_score*100:.2f} %; test_cce_loss: {test_loss:.4f}")
        return
    else:
        if accelerator.is_main_process:
            early_stopping = EarlyStopping(threshold=10, patience=5)

        for epoch in range(args.epochs):
            accelerator.print('Epoch: [%d | %d]' % (epoch + 1, args.epochs))
                
            epoch_s = time.time()

            # import ipdb; ipdb.set_trace()

            # train for one epoch
            train_acc, train_hmn_loss, train_cce_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, accelerator, args)
            accelerator.print(f"[Train] Acc@1: {train_acc:.4f} %, Harmonization Loss: {train_hmn_loss:.4f}, CCE Loss: {train_cce_loss:.4f}")

            # evaluate on validation set
            val_acc, val_cce_loss, alignment_score = evaluate(val_loader, model, criterion, accelerator, args)
            accelerator.print(f"[Val]   Acc@1: {val_acc:.4f} %, Alignment Score: {alignment_score*100:.4f} %, CCE Loss: {val_cce_loss:.4f}")
            
            epoch_e = time.time()
                        
            accelerator.print("Epoch {}: {} secs".format(str(epoch+1), str(int(epoch_e - epoch_s))))

            # Skip warming up stage
            if epoch <= args.warmup: continue 
            
            # save model for best_acc model
            is_best_acc = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            is_best_alignment = alignment_score > best_human_alignment
            best_human_alignment = max(alignment_score, best_human_alignment)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint_accelerator({
                'epoch': epoch + 1,
                "model_name": args.model_name,
                'state_dict': unwrapped_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'alignment': alignment_score,
                'best_human_alignment': best_human_alignment,
                'mode':args.mode
            }, is_best_acc, is_best_alignment, epoch+1, accelerator, args)
            
            accelerator.print("")

            if accelerator.is_main_process and early_stopping(train_acc, val_acc):
                accelerator.print("Early stopping triggered")
                break
                
    if accelerator.is_main_process:
        end_time = time.time()
        accelerator.print('Total hours: ', round((end_time - start_time) / 3600, 2))
        accelerator.print("****************************** DONE! ******************************")
        
        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29501"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    
    # create the command line parser
    parser = argparse.ArgumentParser('Harmonization PyTorch Scripts using Accelerate', add_help=False)
    parser.add_argument("-dd", "--data_dir", required = False, type = str, 
                        default = '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/subset',
                        choices=[
                            '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/full', # small-scale data examples
                            '/media/data_cifs/projects/prj_pseudo_clickme/Dataset/subset'
                        ],
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
                        default = '/media/data_cifs/projects/prj_pseudo_clickme/Checkpoints',
                        help = "please enter a directory save checkpoints")
    parser.add_argument("-bt", "--best_model", required=False, type = str,
                        default = 'best_acc', 
                        choices = [
                            "best_acc",  
                            "best_alignment",
                        ],
                        help="'best_acc', 'best_alignment'?")
    parser.add_argument("-mn", "--model_name", required = False, type = str,
                        default = 'resnet50',
                        help="Please specify a model architecture according to TIMM")
    parser.add_argument("-md", "--mode", required=False, type = str,
                        default = 'imagenet', 
                        choices = [
                            "pseudo",  # pure imagenet images with pseudo clickme maps
                            "mix",     # imagenet images with pseudo and real clickme maps
                            "imagenet", # imagenet images 
                            "clickme",
                        ],
                        help="'pseudo', 'mix' or 'imagenet'?")
    parser.add_argument("-ep", "--epochs", required=False, type = int, default = 3,
                        help="Number of Epochs")
    parser.add_argument("-sp", "--start_epoch", required=False, type = int, default = 0,
                        help="start epoch is usually used with 'resume'")
    parser.add_argument("-bs", "--batch_size", required=False, type = int,default = 4,
                        help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", required=False, type = float, default = 1e-5,
                        help="Learning Rate")
    parser.add_argument("-mt", "--momentum", required=False, type = float, default = 0.9,
                        help="SGD momentum")
    parser.add_argument("-ss", "--step_size", required=False, type = int, default = 25,
                        help="learning rate scheduler")
    parser.add_argument("-gm", "--gamma", required=False, type = float, default = 0.1,
                        help="scheduler parameters, which decides the change of learning rate ")
    parser.add_argument("-wd", "--weight_decay", required=False, type = float, default = 1e-4,
                        help="weight decay, regularization")
    parser.add_argument("-iv", "--interval", required=False, type = int, default = 2,
                        help="Step interval for printing logs")
    parser.add_argument("-nw", "--num_workers", required=False, type = int, default = 8,
                        help="number of workers in dataloader")
    parser.add_argument("-gid", "--gpu_id", required=False, type = int, default = 1,
                        help="specify gpu id for single gpu training")
    parser.add_argument("-tc", "--tpu_cores_per_node", required=False, type = int, default = 1,
                        help="specify the number of tpu cores")
    parser.add_argument("-ckpt", "--ckpt_remain", required=False, type = int, default = 5,
                        help="how many checkpoints can be saved at most?")
    parser.add_argument("-lu", "--logger_update", required=False, type = int, default = 10,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-sd", "--seed", required=False, type = int, default = 42,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-ev", "--evaluate", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to evaluate a model")
    parser.add_argument("-wu", "--warmup", required=False, type = int, default = 5,
                        help="specify warmup epochs, usually <= 5")
    parser.add_argument("-pt", "--pretrained", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to use pretrained model from TIMM")
    parser.add_argument("-rs", "--resume", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to continue (usually used with 'evaluate')")
    parser.add_argument("-gt", "--tpu", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to use Google Cloud Tensor Processing Units")
    parser.add_argument("-wb", "--wandb", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to W&B to record progress")
    
    # modify the configurations according to args parser
    args = parser.parse_args()
    # assert args.interval > 5, f"Please make sure the interval is greater than 5"
    
    run(args)