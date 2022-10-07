""" Reference:
    [1] Hongyi Zhang etc. mixup: BEYOND EMPIRICAL RISK MINIMIZATION. ICLR 2018
    [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    [3] https://www.cnblogs.com/hizhaolei/p/10611141.html
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = 1
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    if use_cuda:
        index = index.cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
