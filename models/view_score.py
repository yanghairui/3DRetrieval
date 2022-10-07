import torch.nn as nn
import torch


class ViewScore(nn.Module):
    def __init__(self, in_channels, views):
        super(ViewScore, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.linear = nn.Linear(views, views)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ x with shape (batch_size, features, views)
        """
        x = self.conv(x)
        x = x.squeeze(1)
        x = self.linear(x)

        return self.softmax(x)


class ViewScore2(nn.Module):
    def __init__(self, in_channels, views):
        super(ViewScore2, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels * 2, out_channels=1, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        # self.bn = nn.BatchNorm1d(256)
        # self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(views, views)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def _sigmoid(self, x):
        """ x with shape (batch_size, views)
        """
        # x_mean = x.mean(dim=1).unsqueeze(1)
        # x = torch.exp(x - x_mean)
        # return x / (torch.exp(-x_mean) + x)
        x = torch.log(torch.abs(x))
        return self.sigmoid(x)

    def forward(self, x):
        """ x with shape (batch_size, features, views)
        """
        # Global
        g_x = x.max(dim=2)[0]
        # Local + Global
        # x = x + g_x.unsqueeze(-1)
        g_x = g_x.unsqueeze(-1)
        x = torch.cat([x, g_x.repeat(1, 1, 12)], 1)

        x = self.conv(x)
        # x_trans = x.permute(2, 0, 1)
        # x = torch.cat([self.relu(self.bn(view).unsqueeze(-1)) for view in x_trans], dim=2)
        # x = self.conv2(x)

        x = x.squeeze(1)
        # x = self.linear(x)

        return self._sigmoid(x)
