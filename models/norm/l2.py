""" x by l2-normalization
"""

import torch
import torch.nn as nn


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()


    def forward(self, x):
        """ x with shape (batch_size, feature_dim)
        or with shape (num_classes, feature_dim)
        """
        l2_norm = torch.norm(x, dim=1)
        l2_norm = l2_norm.unsqueeze(1)

        return x / l2_norm
