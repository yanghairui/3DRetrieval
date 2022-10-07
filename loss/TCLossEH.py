""" Triplet-Center Loss
Reference:
    He et al. Triplet-Center Loss for Multi-View 3D Object Retrieval. CVPR 2018.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class TripletCenterCosineRFLoss(nn.Module):
    def __init__(self, num_classes=21, fea_dim=512, margin=1, use_gpu=True,
                 neg_c2c_margin=1.4,
                 l2Norm=None):
        super(TripletCenterCosineRFLoss, self).__init__()

        self.margin = margin
        self.neg_c2c_margin = neg_c2c_margin
        self.fea_dim = fea_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.l2Norm = l2Norm

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim))

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            norm_centers = self.l2Norm(self.centers)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        batch_size = x.size(0)

        pos_center = 1 - ((x[0] * norm_centers[labels[0]]).sum(dim=0))
        indices = torch.nonzero(torch.ne(classes, labels[0])).view(-1)
        sel_center = torch.index_select(norm_centers, dim=0, index=indices)
        neg_center = torch.min(1 - ((x[0] * sel_center).sum(dim=1)))

        neg_c2c = torch.min((1 - (self.centers[labels[0]] * sel_center).sum(dim=1)))

        loss = F.relu(pos_center + self.margin - neg_center) + \
            F.relu(self.neg_c2c_margin - neg_c2c)

        for batch_idx in range(1, batch_size):
            pos_center = 1 - ((x[batch_idx] * norm_centers[labels[batch_idx]]).sum(dim=0))
            indices = torch.nonzero(torch.ne(classes, labels[batch_idx])).view(-1)
            sel_center = torch.index_select(norm_centers, dim=0, index=indices)
            neg_center = torch.min(1 - ((x[batch_idx] * sel_center).sum(dim=1)))

            neg_c2c = torch.min((1 - (self.centers[labels[batch_idx]] * sel_center).sum(dim=1)))

            loss += F.relu(pos_center + self.margin - neg_center) + \
                F.relu(self.neg_c2c_margin - neg_c2c)

        return loss / batch_size


class TripletCenterLoss(nn.Module):
    """ Triplet-Center Loss enhance edition.
    Optimizing the distance between centers.
    """
    def __init__(self, num_classes=21, fea_dim=512, margin=5, use_gpu=True):
        super(TripletCenterLoss, self).__init__()

        self.margin = margin
        self.fea_dim = fea_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu

        if use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.fea_dim))

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim)
            labels with shape (batch_size)
        """
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        batch_size = x.size(0)

        pos_center = torch.pow(x[0] - self.centers[labels[0]], 2).sum() / 2.
        indices = torch.nonzero(torch.ne(classes, labels[0])).view(-1)
        sel_fea = torch.index_select(self.centers, dim=0, index=indices)
        neg_center = torch.min(torch.pow(x[0] - sel_fea, 2).sum(dim=1)) / 2.

        centers = torch.min(torch.pow(self.centers[labels[0]] - sel_fea, 2).sum(dim=1)) / 2.

        loss = F.relu(pos_center + self.margin - neg_center) + F.relu(self.margin + 2 - centers)

        for batch_indx in range(1, batch_size):
            pos_center = torch.pow(x[batch_indx] - self.centers[labels[batch_indx]], 2).sum() / 2.
            indices = torch.nonzero(torch.ne(classes, labels[batch_indx])).view(-1)
            sel_fea = torch.index_select(self.centers, dim=0, index=indices)
            neg_center = torch.min(torch.pow(x[batch_indx] - sel_fea, 2).sum(dim=1)) / 2.

            centers = torch.min(torch.pow(self.centers[labels[batch_indx]] - sel_fea, 2).sum(dim=1)) / 2.

            loss += F.relu(pos_center + self.margin - neg_center) + F.relu(self.margin + 2 - centers)

        return loss / batch_size
