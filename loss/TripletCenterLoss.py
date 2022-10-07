import os

import torch.nn as nn
import torch
import torch.nn.functional as F


class TripletCenterCosineLoss(nn.Module):
    def __init__(self, config=None, num_classes=90, fea_dim=512, margin=1, use_gpu=True,
                 l2Norm=None):
        super(TripletCenterCosineLoss, self).__init__()

        self.margin = margin
        self.fea_dim = fea_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.l2Norm = l2Norm

        # self.centers = torch.load(os.path.join(config.PATH, 'class_centers.pth'))

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn((self.num_classes, self.fea_dim)).cuda())
        else:
            self.centers = nn.Parameter(torch.randn((self.num_classes, self.fea_dim)))

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

        loss = F.relu(pos_center + self.margin - neg_center)

        for batch_idx in range(1, batch_size):
            pos_center = 1 - ((x[batch_idx] * norm_centers[labels[batch_idx]]).sum(dim=0))
            indices = torch.nonzero(torch.ne(classes, labels[batch_idx])).view(-1)
            sel_center = torch.index_select(norm_centers, dim=0, index=indices)
            neg_center = torch.min(1 - ((x[batch_idx] * sel_center).sum(dim=1)))

            loss += F.relu(pos_center + self.margin - neg_center)

        return loss / batch_size


class TripletCenterAbsCosineLoss(nn.Module):
    def __init__(self, config=None, num_classes=90, fea_dim=512, margin=1, use_gpu=True,
                 l2Norm=None):
        super(TripletCenterAbsCosineLoss, self).__init__()

        self.margin = margin
        self.fea_dim = fea_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.l2Norm = l2Norm

        # self.centers = torch.load(os.path.join(config.PATH, 'class_centers.pth'))

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn((self.num_classes, self.fea_dim)).cuda())
        else:
            self.centers = nn.Parameter(torch.randn((self.num_classes, self.fea_dim)))

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

        loss = F.relu(pos_center + self.margin - neg_center) + pos_center

        for batch_idx in range(1, batch_size):
            pos_center = 1 - ((x[batch_idx] * norm_centers[labels[batch_idx]]).sum(dim=0))
            indices = torch.nonzero(torch.ne(classes, labels[batch_idx])).view(-1)
            sel_center = torch.index_select(norm_centers, dim=0, index=indices)
            neg_center = torch.min(1 - ((x[batch_idx] * sel_center).sum(dim=1)))

            loss += (F.relu(pos_center + self.margin - neg_center) + pos_center)

        return loss / batch_size


class TripletCenterLoss(nn.Module):
    def __init__(self, num_classes=90, fea_dim=512, margin=5, use_gpu=True):
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

        loss = F.relu(pos_center + self.margin - neg_center)

        for batch_indx in range(1, batch_size):
            pos_center = torch.pow(x[batch_indx] - self.centers[labels[batch_indx]], 2).sum() / 2.
            indices = torch.nonzero(torch.ne(classes, labels[batch_indx])).view(-1)
            sel_fea = torch.index_select(self.centers, dim=0, index=indices)
            neg_center = torch.min(torch.pow(x[batch_indx] - sel_fea, 2).sum(dim=1)) / 2.

            loss += F.relu(pos_center + self.margin - neg_center)

        return loss / batch_size


class TripletCenterLossV2(nn.Module):
    """ Replace max(x, 0) by log(1 + e^x)
    """
    def __init__(self, num_classes=21, fea_dim=512, margin=5, use_gpu=True):
        super(TripletCenterLossV2, self).__init__()

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

        loss = torch.log(1 + torch.exp(pos_center + self.margin - neg_center))

        for batch_indx in range(1, batch_size):
            pos_center = torch.pow(x[batch_indx] - self.centers[labels[batch_indx]], 2).sum() / 2.
            indices = torch.nonzero(torch.ne(classes, labels[batch_indx])).view(-1)
            sel_fea = torch.index_select(self.centers, dim=0, index=indices)
            neg_center = torch.min(torch.pow(x[batch_indx] - sel_fea, 2).sum(dim=1)) / 2.

            loss += torch.log(1 + torch.exp(pos_center + self.margin - neg_center))

        return loss / batch_size
