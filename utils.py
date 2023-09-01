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

    return human_alignment

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
    