""" Triplet-Centor Loss without parameters.
"""

import os
import torch.nn as nn
import torch.nn.functional as F
import torch


class TCLWithoutParametersLoss(nn.Module):
    def __init__(self, base_dir='./data', margin=5,
                 num_classes=21, use_gpu=True):
        super(TCLWithoutParametersLoss, self).__init__()

        self.num_classes = num_classes
        self.margin = margin
        self.use_gpu = use_gpu

        # (num_classes, feature_dim)
        self.centers = torch.load(
            os.path.join(base_dir, 'centers.pth'))
        self.centers = self.centers['centers'].data

    def forward(self, features, labels):
        """ features with shape (batch_size, feature_dim)
            labels with shape (batch_size)
        """
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        batch_size = features.size(0)

        pos_center = torch.pow(
            features[0] - self.centers[labels[0]], 2).sum() / 2.
        indices = torch.nonzero(torch.ne(classes, labels[0])).view(-1)
        sel_fea = torch.index_select(self.centers, dim=0, index=indices)
        neg_center = torch.min(
            torch.pow(features[0] - sel_fea, 2).sum(dim=1)) / 2.

        loss = F.relu(pos_center + self.margin - neg_center)

        for batch_idx in range(1, batch_size):
            pos_center = torch.pow(
                features[batch_idx] - self.centers[labels[batch_idx]], 2).sum() / 2.
            indices = torch.nonzero(torch.ne(classes, labels[batch_idx])).view(-1)
            sel_fea = torch.index_select(self.centers, dim=0, index=indices)
            neg_center = torch.min(
                torch.pow(features[batch_idx] - sel_fea, 2).sum(dim=1)) / 2.

            loss += F.relu(pos_center + self.margin - neg_center)

        return loss / batch_size
