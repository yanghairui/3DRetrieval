"""Center Loss
Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
"""

import torch.nn as nn
import torch

class CenterLoss(nn.Module):
    def __init__(self, num_classes=21, fea_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()

        self.fea_dim = fea_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim))

    def forward(self, x, labels):
        """ X with shape (batch_size, feature_dim)
            labels with shape (batch_size)
        """
        batch_size = x.size(0)

        loss = torch.pow(x[0] - self.centers[labels[0]], 2).sum()

        for batch_idx in range(1, batch_size):
            loss += torch.pow(x[batch_idx] - self.centers[labels[batch_idx]], 2).sum()

        loss /= 2.

        return loss / batch_size
