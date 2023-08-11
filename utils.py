import torch
import numpy as np
from scipy.stats import spearmanr

from configs import DefaultConfigs

import torch.distributed as dist

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

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def synchronize_between_processes(self, isXLA):
        for meter in self.meters.values():
            meter.synchronize_between_processes(isXLA)

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
            return
        if not is_dist_avail_and_initialized(isXLA):
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / self.count
    
def spearman_correlation(heatmaps_a, heatmaps_b):
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

def compute_human_alignment(predicted_heatmaps, clickme_heatmaps):
    HUMAN_SPEARMAN_CEILING = 0.65753

    if len(clickme_heatmaps.shape) == 4:
        clickme_heatmaps = clickme_heatmaps[:, 0, :, :]
    if len(predicted_heatmaps.shape) == 4:
        predicted_heatmaps = predicted_heatmaps[:, 0, :, :]

    scores = spearman_correlation(predicted_heatmaps, clickme_heatmaps)
    human_alignment = np.mean(scores) / HUMAN_SPEARMAN_CEILING

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

def is_main_process():
    return get_rank() == 0

def is_dist_avail_and_initialized(isXLA):
    if isXLA:
        raise Exception("This function should not be called in XLA")
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

