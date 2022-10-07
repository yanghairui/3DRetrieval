""" Linear transform
"""

import torch.nn as nn


class LinearTransform(nn.Module):
    def __init__(self, in_feats, out_feats, expansion=1, dropout=False,
                num_classes=90):
        super(LinearTransform, self).__init__()

        self.expansion = expansion
        self.dropout = dropout

        self.linear1 = nn.Linear(in_feats, in_feats)
        self.bn1 = nn.BatchNorm1d(in_feats)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        if self.dropout:
            self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_feats, out_feats)

        self.fc = nn.Linear(out_feats, num_classes)

    def forward(self, x, norm=None):
        x = self.linear1(x)
        x = self.leakyrelu(self.bn1(x))
        if self.dropout:
             x = self.drop1(x)
        x = self.linear2(x)

        feats = x
        x = self.fc(feats)
        if norm is not None:
            feats = norm(feats)

        return feats, x
