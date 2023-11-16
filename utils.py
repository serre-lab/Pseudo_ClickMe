import os
import argparse
import numpy as np
import pathlib
import builtins
import datetime
import torch
import torch.distributed as dist
from torchmetrics.regression import SpearmanCorrCoef
from scipy.stats import spearmanr

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, isXLA):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if isXLA:
            xm.master_print('  '.join(entries))
        else:
            print('  '.join(entries))
        return 

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def synchronize_between_processes(self, isXLA):
        for meter in self.meters:
            meter.synchronize_between_processes(isXLA)
        return

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def synchronize_between_processes(self, isXLA):
        if isXLA:
            t = torch.tensor([self.count, self.sum], device=xm.xla_device())
            t = xm.all_reduce(xm.REDUCE_SUM, t).tolist()
            self.count = int(t[0])
            self.sum = t[1]
            # self.avg = self.sum / self.count
            return
        if not is_dist_avail_and_initialized(isXLA):
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        # self.avg = self.sum / self.count
        return
    
def spearman_correlation_np(heatmaps_a, heatmaps_b):
    """
    Computes the Spearman correlation between two sets of heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    spearman_correlations
        Array of Spearman correlation score between the two sets of heatmaps.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []

    try:
        heatmaps_a = heatmaps_a.numpy()
        heatmaps_b = heatmaps_b.numpy()
    except:
        heatmaps_a = heatmaps_a.cpu().numpy()
        heatmaps_b = heatmaps_b.cpu().numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        rho, _ = spearmanr(ha.flatten(), hb.flatten())
        scores.append(rho)

    return np.array(scores)

def spearman_correlation(heatmaps_a, heatmaps_b):
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []
    batch_size = heatmaps_a.shape[0]
    spearman = SpearmanCorrCoef(num_outputs=1)
    heatmaps_a, heatmaps_b = heatmaps_a.reshape(batch_size, -1), heatmaps_b.reshape(batch_size, -1)
    for i in range(batch_size):
        score = spearman(heatmaps_a[i, :], heatmaps_b[i, :])
        scores.append(score)

    return torch.tensor(scores)

def compute_human_alignment(predicted_heatmaps, clickme_heatmaps):
    HUMAN_SPEARMAN_CEILING = 0.65753

    if len(clickme_heatmaps.shape) == 4:
        clickme_heatmaps = clickme_heatmaps[:, 0, :, :]
    if len(predicted_heatmaps.shape) == 4:
        predicted_heatmaps = predicted_heatmaps[:, 0, :, :]

    scores = spearman_correlation(predicted_heatmaps, clickme_heatmaps)
    human_alignment = scores.mean() / HUMAN_SPEARMAN_CEILING

    return human_alignment.to(clickme_heatmaps.device)

def get_world_size(isXLA):
    if isXLA:
        return xm.xrt_world_size()
    if not is_dist_avail_and_initialized(isXLA):
        return 1
    return torch.distributed.get_world_size()

def get_rank(isXLA):
    if isXLA:
        return xm.get_ordinal()
    if not is_dist_avail_and_initialized(isXLA):
        return 0
    return torch.distributed.get_rank()

def is_main_process(isXLA):
    return get_rank(isXLA) == 0

def is_dist_avail_and_initialized(isXLA):
    if isXLA:
        raise Exception("This function should not be called in XLA")
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

class str2bool(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ('true', 't', '1'):
            setattr(namespace, self.dest, True)
        elif values.lower() in ('false', 'f', '0'):
            setattr(namespace, self.dest, False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for {self.dest}: {values}")
        
def init_distributed_mode(args):
    print("Init distributed mode")
    if args.tpu:
        is_master = xm.get_ordinal()
        setup_for_distributed(args, is_master==0)
        return
    
def setup_for_distributed(args, is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print
    isTPU = args.tpu
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if args == ('torch_xla.core.xla_model::mark_step',):
            # XLA server step tracking
            if is_master:
                builtin_print(*args, **kwargs)
            return
        force = force or (not isTPU and get_world_size(args.tpu) > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
  
def save_model(args, state, filename):
    if args.tpu:
        xm.save(state, filename, global_master=True) # save ckpt on master process
    else: 
        torch.save(state, filename)
       
def save_checkpoint(state, is_best_acc, is_best_alignment, epoch, global_rank, args):
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
     
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True) # "/mnt/disks/bucket/pseudo_clickme/"
        
    model_dir = os.path.join(args.weights, args.model_name) # "/mnt/disks/bucket/pseudo_clickme/resnet50"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        
    save_dir = os.path.join(model_dir, args.mode) # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.join(save_dir, "ckpt_" + str(epoch) + ".pth.tar")
    save_model(args, state, filename)
 
    if is_best_acc:
        best_filename = os.path.join(save_dir, 'best_acc.pth.tar') # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/best_acc.pth"
        save_model(args, state, best_filename)
        if args.tpu:
            xm.master_print("The best_acc model is saved at EPOCH", str(epoch))
        else:
            print("The best_acc model is saved at EPOCH", str(epoch))
            
    if is_best_alignment:
        best_filename = os.path.join(save_dir, 'best_alignment.pth.tar') # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/best_acc.pth"
        save_model(args, state, best_filename)
        if args.tpu:
            xm.master_print("The best_acc model is saved at EPOCH", str(epoch))
        else:
            print("The best_alignment model is saved at EPOCH", str(epoch))
        
    rmfile = os.path.join(save_dir, "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
    if global_rank == 0 and os.path.exists(rmfile):
        os.remove(rmfile)
        if args.tpu:
            xm.master_print("Removed ", "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
        else:
            print("Removed ", "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
            
def save_checkpoint_accelerator(state, is_best_acc, is_best_alignment, epoch, accelerator, args):
    '''
    /Checkpoints/
    |__resnet50
    |    |__imagenet
    |    |    |__ckpt_0.pth
    |    |    |__best.pth
    |    |__mix
    |    |__pseudo
    |...
    ''' 
     
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True) # "/Checkpoints/"
        
    model_dir = os.path.join(args.weights, args.model_name) # "/Checkpoints/resnet50"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        
    save_dir = os.path.join(model_dir, args.mode) # "Checkpoints/resnet50/imagenet/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.join(save_dir, "ckpt_" + str(epoch) + ".pth.tar")
    accelerator.save(state, filename)
 
    if is_best_acc:
        best_filename = os.path.join(save_dir, 'best_acc.pth.tar') # "Checkpoints/resnet50/imagenet/best_acc.pth"
        accelerator.save(state, best_filename)
        accelerator.print("The best_acc model is saved at EPOCH", str(epoch))
            
    if is_best_alignment:
        best_filename = os.path.join(save_dir, 'best_alignment.pth.tar') # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/best_acc.pth"
        accelerator.save(state, best_filename)
        accelerator.print("The best_alignment model is saved at EPOCH", str(epoch))
        
    rmfile = os.path.join(save_dir, "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
    if accelerator.is_main_process and os.path.exists(rmfile):
        os.remove(rmfile)
        accelerator.print("Removed ", "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
        
class ProgressMeterAcc(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, accelerator):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if accelerator:
            accelerator.print('  '.join(entries))
        return 

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def synchronize_between_processes(self, accelerator):
        for meter in self.meters:
            meter.synchronize_between_processes(accelerator)
        return

class AverageMeterAcc(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def synchronize_between_processes(self, accelerator):
        c = accelerator.reduce(torch.tensor(self.count, dtype=torch.float32).to(accelerator.device), reduction="mean")
        s = accelerator.reduce(torch.tensor(self.sum, dtype=torch.float32).to(accelerator.device), reduction="mean")
        self.count = int(c.item())
        self.sum = s.item()
        self.avg = self.sum / self.count
        return
        
        # t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        # dist.barrier()
        # dist.all_reduce(t)
        # t = t.tolist()
        # self.count = int(t[0])
        # self.sum = t[1]
        # # self.avg = self.sum / self.count
        # return

import math
from typing import Union, List
import torch
from torch.optim.lr_scheduler import _LRScheduler

'''
https://github.com/santurini/cosine-annealing-linear-warmup/tree/main
'''
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: List[float] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
                 ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow is not None:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs is not None:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(self.max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(self.min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class EarlyStopping:
    def __init__(self, threshold=10, patience=5):
        """
        Initializes the early stopping mechanism.
        :param threshold: The maximum allowed difference between train and test accuracy.
        :param patience: How many epochs to wait after the threshold is first exceeded.
        """
        self.threshold = threshold
        self.patience = patience
        self.patience_counter = 0
        self.best_diff = float('inf')

    def __call__(self, train_acc, test_acc):
        """
        Call this at the end of each epoch, providing the current train and test accuracies.
        :param train_acc: Training accuracy for the current epoch.
        :param test_acc: Testing/validation accuracy for the current epoch.
        :return: True if training should be stopped, False otherwise.
        """
        current_diff = abs(train_acc - test_acc)

        if current_diff < self.best_diff:
            self.best_diff = current_diff
            self.patience_counter = 0
        elif current_diff > self.threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True  # Stop training

        return False  # Continue training

def compute_gradient_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def compute_image_gradient_norm(images_grad, norm_type=2):
    # Compute the L2 norm for each image in the batch individually
    individual_norms = torch.norm(images_grad.view(images_grad.shape[0], -1), p=norm_type, dim=1)

    # Compute the average L2 norm for the batch
    average_norm = torch.mean(individual_norms)

    return average_norm

# Gradient Flow
import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters, filename):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
#         p = p.detach().cpu()
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
#     plt.figure(figsize=(15, 15))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
#     plt.ylim(ymin=0, ymax=5)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('../'+str(filename)+'.png')
    plt.clf()



