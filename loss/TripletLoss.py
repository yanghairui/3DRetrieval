import torch.nn.functional as F
import torch
import torch.nn as nn


__all__ = ['TripletLoss', 'TripletLoss3D']

class TripletLoss(nn.Module):
    def __init__(self, margin=1.5, metric='euc', norm=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        self.loss = nn.TripletMarginLoss(margin)
        self.norm = norm

    def forward(self, feats, pos_negs):
        """
        Args:
            x (batch_size, num_attributes)
            y (batch_size, 1)
        """
        x = feats
        if self.norm:
            x = self.norm(x)
        hardest_pos, hardest_neg = pos_negs[0]
        dist_pos = torch.pow(x[0] - hardest_pos, 2).sum()
        dist_neg = torch.pow(x[0] - hardest_neg, 2).sum()
        loss = F.relu(dist_pos + self.margin - dist_neg)
        # anchors = x[0].unsqueeze(0)
        # positives = hardest_pos.unsqueeze(0)
        # negtives = hardest_neg.unsqueeze(0)
        # loss = self.loss(x[0], hardest_pos, hardest_neg)

        for idx in range(1, x.size(0)):
            hardest_pos, hardest_neg = pos_negs[idx]
            dist_pos = torch.pow(x[idx] - hardest_pos, 2).sum()
            dist_neg = torch.pow(x[idx] - hardest_neg, 2).sum()
            loss += F.relu(dist_pos + self.margin - dist_neg)
            # anchors = torch.cat([anchors, x[idx].unsqueeze(0)], 0)
            # positives = torch.cat([positives, hardest_pos.unsqueeze(0)], 0)
            # negtives = torch.cat([negtives, hardest_neg.unsqueeze(0)], 0)

        # return self.loss(anchors, positives, negtives)
        return loss / x.size(0)


class TripletLossJoint(nn.Module):
    def __init__(self, margin=1.5, metric='euc', norm=None):
        super(TripletLossJoint, self).__init__()
        self.margin = margin
        self.metric = metric
        self.loss = nn.TripletMarginLoss(margin)
        self.norm = norm

    def forward(self, feats, pos, neg):
        """
        Args:
            x (batch_size, num_attributes)
            y (batch_size, 1)
        """
        x = feats
        if self.norm:
            x = self.norm(x)
        hardest_pos = pos[0]
        hardest_neg = neg[0]
        dist_pos = torch.pow(x[0] - hardest_pos, 2).sum()
        dist_neg = torch.pow(x[0] - hardest_neg, 2).sum()
        loss = F.relu(dist_pos + self.margin - dist_neg)
        # anchors = x[0].unsqueeze(0)
        # positives = hardest_pos.unsqueeze(0)
        # negtives = hardest_neg.unsqueeze(0)
        # loss = self.loss(x[0], hardest_pos, hardest_neg)

        for idx in range(1, x.size(0)):
            hardest_pos = pos[idx]
            hardest_neg = neg[idx]
            dist_pos = torch.pow(x[idx] - hardest_pos, 2).sum()
            dist_neg = torch.pow(x[idx] - hardest_neg, 2).sum()
            loss += F.relu(dist_pos + self.margin - dist_neg)
            # anchors = torch.cat([anchors, x[idx].unsqueeze(0)], 0)
            # positives = torch.cat([positives, hardest_pos.unsqueeze(0)], 0)
            # negtives = torch.cat([negtives, hardest_neg.unsqueeze(0)], 0)

        # return self.loss(anchors, positives, negtives)
        return loss / x.size(0)


class TripletLoss3D(nn.Module):
    def __init__(self, margin=1.0, metric='euc', norm=None):
        super(TripletLoss3D, self).__init__()
        self.margin = margin
        self.metric = metric
        self.loss = nn.TripletMarginLoss(margin)
        self.norm = norm

    def _online_example_mining(self, idx, x, y):
        """
        Args:
            anchor (num_attributes)
            anchor_idx (1)
            x (batch_size, num_attributes)
            y (batch_size, 1)
        Returns:
            tuple (2): positive, negtive
        """
        pos_class_mask = (y == y[idx]).squeeze()
        neg_class_mask = (y != y[idx]).squeeze()
        pos_x = x[pos_class_mask]
        neg_x = x[neg_class_mask]

        # Select hardest positive example
        pos_dist = torch.pow(x[idx] - pos_x, 2).sum(dim=1)
        pos_indx = torch.argmax(pos_dist)
        hardest_pos = pos_x[pos_indx]
        # Select hardest negtive example
        neg_dist = torch.pow(x[idx] - neg_x, 2).sum(dim=1)
        neg_indx = torch.argmin(neg_dist)
        hardest_neg = neg_x[neg_indx]

        return hardest_pos, hardest_neg

    def forward(self, x, y):
        """
        Args:
            x (batch_size, num_attributes)
            y (batch_size, 1)
        """
        # Normalize
        if self.norm:
            x = self.norm(x)
        hardest_pos, hardest_neg = self._online_example_mining(0, x, y)
        dist_pos = torch.pow(x[0] - hardest_pos, 2).sum()
        dist_neg = torch.pow(x[0] - hardest_neg, 2).sum()
        loss = F.relu(dist_pos + self.margin - dist_neg)
        # anchors = x[0].unsqueeze(0)
        # positives = hardest_pos.unsqueeze(0)
        # negtives = hardest_neg.unsqueeze(0)
        # loss = self.loss(x[0], hardest_pos, hardest_neg)

        for idx in range(1, x.size(0)):
            hardest_pos, hardest_neg = self._online_example_mining(idx, x, y)
            dist_pos = torch.pow(x[idx] - hardest_pos, 2).sum()
            dist_neg = torch.pow(x[idx] - hardest_neg, 2).sum()
            loss += F.relu(dist_pos + self.margin - dist_neg)
            # anchors = torch.cat([anchors, x[idx].unsqueeze(0)], 0)
            # positives = torch.cat([positives, hardest_pos.unsqueeze(0)], 0)
            # negtives = torch.cat([negtives, hardest_neg.unsqueeze(0)], 0)

        # return self.loss(anchors, positives, negtives)
        return loss / x.size(0)