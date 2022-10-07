import torch.optim as optim

def CosineAnnealing(optimizer, max_iters):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)
