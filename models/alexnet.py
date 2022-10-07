import torch.nn as nn
import torch

try:  # torch.__version__ <= 0.4.1
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
except:  # torch.__version >= 1.0
    from torchvision.models.utils import load_state_dict_from_url

from models import view_score

from . import lt as LT

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=90, view_ensemble=False, num_views=12):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.view_ensemble = view_ensemble
        if self.view_ensemble:
            self.view_select = view_score.ViewScore2(256 * 6 * 6, num_views)

        # self.lt = LT.LinearTransform(4096, 4096,
        #                              num_classes=num_classes)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        self.classify = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, norm):
        """ x with shape (B, 12, C, H, W)"""

        # Swap batch and views dims
        x = x.transpose(0, 1)

        # View pool
        view_pool = []
        for v in x:
            v = self.features(v)
            v = self.avgpool(v).view(v.size(0), -1)
            view_pool.append(v)

        if self.view_ensemble:
            x = torch.cat([view.unsqueeze(2) for view in view_pool], dim=2)
            v_score = self.view_select(x).unsqueeze(1)
            # print(v_score[0])
            x = x * (0 + v_score)
            # pooled_view = x.sum(dim=2)
            # pooled_view = x.max(dim=2)[0]

            # v_score = self.view_select2(x).unsqueeze(1)
            # x = x * (1 + v_score)

            pooled_view = x.max(dim=2)[0]
        else:
            pooled_view = view_pool[0]
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

        features = self.classifier(pooled_view)
        if norm:
            features = norm(features)
        class_features = self.classify(features)

        return features, class_features


class AlexNetCls(nn.Module):
    def __init__(self, num_classes=90):
        super(AlexNetCls, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        self.classify = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """ x with shape (B, C, H, W)"""
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        features = self.classifier(x)
        class_features = self.classify(features)

        return features, class_features


def alexnet(pretrained=False, progress=True, cls=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if cls:
        model = AlexNetCls(**kwargs)
    else:
        model = AlexNet(**kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    net = alexnet(pretrained=False, cls=True)
    x = torch.randn(2, 3, 224, 224)
    features, class_features = net(x)
    print(features.size(), class_features.size())
