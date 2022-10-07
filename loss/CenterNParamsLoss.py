""" The closer the feature is to the center, the better
"""

import os

import torch
import torch.nn as nn


class CenterWoParamsCosineLoss(nn.Module):
    def __init__(self, center_path='./data/alexnet.pth', l2Norm=None):
        super(CenterWoParamsCosineLoss, self).__init__()
        print("CenterWoParamsCosineLoss")
        self.centers = torch.load(
            # os.path.join(center_path, 'refined_centers_14.pth')
            center_path
        )['CENTERS']['centers'].data

        # Normalization
        self.l2Norm = l2Norm
        if l2Norm:
            self.centers = l2Norm(self.centers)

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            x = self.l2Norm(x)

        batch_size = x.size(0)

        loss = 1 - (x[0] * self.centers[labels[0]]).sum(dim=0)

        for batch_idx in range(1, batch_size):
            loss += (1 - (x[batch_idx] * self.centers[labels[batch_idx]]).sum(dim=0))

        return loss / batch_size


class CenterWoParamMultiCosineSoftmaxLoss(nn.Module):
    def __init__(self, centers_path, l2Norm=None, num_classes=90, use_cuda=True):
        super(CenterWoParamMultiCosineSoftmaxLoss, self).__init__()
        self.l2Norm = l2Norm
        self.centers = []
        for class_idx in range(num_classes):
            center = torch.load(os.path.join(centers_path, '{}.centers'.format(class_idx)))
            # Normalization
            if self.l2Norm:
                center = self.l2Norm(center)
            if use_cuda:
                center = center.cuda()
            self.centers.append(center)

    def _forward(self, x, class_idx):
        dst = 1 - (x * self.centers[class_idx]).sum(dim=1)
        score = (2 - dst).softmax(dim=0)
        return (score * dst).sum()

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            x = self.l2Norm(x)

        batch_size = x.size(0)
        loss = self._forward(x[0], labels[0])
        for batch_idx in range(1, batch_size):
            loss += self._forward(x[batch_idx], labels[batch_idx])

        return loss / batch_size


class CenterWoParamMultiCosineLoss(nn.Module):
    def __init__(self, centers_path, l2Norm=None, num_classes=90, use_cuda=True):
        super(CenterWoParamMultiCosineLoss, self).__init__()
        self.l2Norm = l2Norm
        self.centers = []
        for class_idx in range(num_classes):
            center = torch.load(os.path.join(centers_path, '{}.centers'.format(class_idx)))
            # Normalization
            if self.l2Norm is not None:
                center = self.l2Norm(center)
            if use_cuda:
                center = center.cuda()
            self.centers.append(center)

    def _forward(self, x, class_idx):
        inner_dsts = []
        for center_idx in range(self.centers[class_idx].size(0)):
            if self.l2Norm is not None:
                inner_dsts.append(1 - (x * self.centers[class_idx][center_idx]).sum(dim=0))
            else:
                inner_dsts.append(torch.pow(x - self.centers[class_idx][center_idx], 2).sum(dim=0))

        detach_inner_dsts = [item.detach() for item in inner_dsts]
        sum_inner_dsts = sum(detach_inner_dsts)
        inner_score = torch.Tensor(list(map(lambda y: y / sum_inner_dsts, detach_inner_dsts))).cuda()

        if inner_score.size(0) == 1:
            sub_loss = inner_score[0] * inner_dsts[0]
        else:
            if self.l2Norm is not None:
                sub_loss = (1 - inner_score[0]) * inner_dsts[0]
            else:
                sub_loss = inner_score[0] * inner_dsts[0]
        for inner_idx in range(1, inner_score.size(0)):
            if self.l2Norm is not None:
                sub_loss += (1 - inner_score[inner_idx]) * inner_dsts[inner_idx]
            else:
                sub_loss += inner_score[inner_idx] * inner_dsts[inner_idx]

        return sub_loss

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            x = self.l2Norm(x)

        batch_size = x.size(0)
        loss = self._forward(x[0], labels[0])
        for batch_idx in range(1, batch_size):
            loss += self._forward(x[batch_idx], labels[batch_idx])

        return loss / batch_size


class CenterWoParamMultiCosineLossV2(nn.Module):
    def __init__(self, centers_path, l2Norm=None, num_classes=90, use_cuda=True):
        super(CenterWoParamMultiCosineLossV2, self).__init__()
        self.l2Norm = l2Norm
        self.centers = []
        for class_idx in range(num_classes):
            center = torch.load(os.path.join(centers_path, '{}.centers'.format(class_idx)))
            # Normalization
            if self.l2Norm:
                center = self.l2Norm(center)
            if use_cuda:
                center = center.cuda()
            self.centers.append(center)

    def _forward(self, x, class_idx):
        inner_dsts = []
        for center_idx in range(self.centers[class_idx].size(0)):
            inner_dsts.append(1 + (x * self.centers[class_idx][center_idx]).sum(dim=0))

        detach_inner_dsts = [item.detach() for item in inner_dsts]
        sum_inner_dsts = sum(detach_inner_dsts)
        inner_score = torch.Tensor(list(map(lambda y: y / sum_inner_dsts, detach_inner_dsts))).cuda()

        sub_loss = inner_score[0] * inner_dsts[0]
        for inner_idx in range(1, inner_score.size(0)):
            sub_loss += inner_score[inner_idx] * inner_dsts[inner_idx]

        return sub_loss

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            x = self.l2Norm(x)

        batch_size = x.size(0)
        loss = self._forward(x[0], labels[0])
        for batch_idx in range(1, batch_size):
            loss += self._forward(x[batch_idx], labels[batch_idx])

        return loss / batch_size


class CenterWoParamMultiCosineNearLoss(nn.Module):
    def __init__(self, centers_path, l2Norm=None, num_classes=90, use_cuda=True):
        super(CenterWoParamMultiCosineNearLoss, self).__init__()
        self.l2Norm = l2Norm
        self.centers = []
        for class_idx in range(num_classes):
            center = torch.load(os.path.join(centers_path, '{}.centers'.format(class_idx)))
            # Normalization
            if self.l2Norm:
                center = self.l2Norm(center)
            if use_cuda:
                center = center.cuda()
            self.centers.append(center)

    def _forward(self, x, class_idx):
        inner_dsts = []
        for center_idx in range(self.centers[class_idx].size(0)):
            inner_dsts.append(1 - (x * self.centers[class_idx][center_idx]).sum(dim=0))

        detach_inner_dsts = torch.FloatTensor(inner_dsts)
        min_val_idx = torch.argmin(torch.Tensor(detach_inner_dsts))
        subclasses = torch.arange(detach_inner_dsts.size(0)).long()
        indices = torch.nonzero(torch.ne(subclasses, min_val_idx)).view(-1)
        valid_dsts = torch.index_select(torch.FloatTensor(inner_dsts), dim=0, index=indices)

        return (inner_dsts[min_val_idx] / detach_inner_dsts.sum()) * inner_dsts[min_val_idx] + ((1 - (valid_dsts / detach_inner_dsts.sum())) * valid_dsts).sum()

        # return inner_dsts[min_val_idx], valid_dsts

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim) and ||x||2 = 1
            labels with shape (batch_size)
        """
        if self.l2Norm:
            x = self.l2Norm(x)

        batch_size = x.size(0)
        loss = self._forward(x[0], labels[0])
        for batch_idx in range(1, batch_size):
            loss += self._forward(x[batch_idx], labels[batch_idx])

        return loss / batch_size


class CenterWoParamsLoss(nn.Module):
    def __init__(self, center_path='./data'):
        super(CenterWoParamsLoss, self).__init__()

        self.centers = torch.load(
            os.path.join(center_path, 'ResNet50mAP_5and15.pth')
        )['CENTERS']['centers'].data

    def forward(self, x, labels):
        """ x with shape (batch_size, feature_dim)
            labels with shape (batch_size)
        """
        batch_size = x.size(0)

        loss = torch.pow(x[0] - self.centers[labels[0]], 2).sum()

        for batch_idx in range(1, batch_size):
            loss += torch.pow(x[batch_idx] - self.centers[labels[batch_idx]], 2).sum()

        loss /= 2.

        return loss / batch_size
