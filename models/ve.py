import torch.nn as nn
import torch


class ViewEnsemble(nn.Module):
    def __init__(self, num_views=12, use_gpu=True):
        super(ViewEnsemble, self).__init__()

        self.num_views = num_views
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.weight = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(
                        self.num_views, 1)).cuda())
        else:
            self.weight = nn.Parameter(
                nn.init.xavier_normal_(
                    torch.empty(
                        self.num_views, 1)))

    def forward(self, features):
        """ features with shape (batch_size, feature_dim, num_views)
            Return: (batch_size, feature_dim)
        """
        batch_size = features.size(0)
        batch_weight = self.weight.unsqueeze(0).repeat(
            batch_size, 1, 1)
        return torch.bmm(features, batch_weight).squeeze(-1)
