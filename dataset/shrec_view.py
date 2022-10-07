import torch
import os

import torch.utils.data as data
from PIL import Image

class ViewSet(data.Dataset):
    def __init__(self, phase='train', base_dir='./data',
                 transform=None):
        super(ViewSet, self).__init__()

        self.transform = transform
        self.phase = phase
        assert self.phase in ('train', 'val', 'test')
        if self.phase == 'train':
            self.path = os.path.join(
                base_dir, 'split', 'shrec2019_View_train.txt')
            self.keep = os.path.join(base_dir, 'train.txt')
        elif self.phase == 'val':
            self.path = os.path.join(
                base_dir, 'split', 'shrec2019_View_train.txt')
            self.keep = os.path.join(base_dir, 'val.txt')
        else:
            self.path = os.path.join(
                base_dir, 'split', 'shrec2019_View_test.txt')

        if hasattr(self, 'keep'):
            with open(self.keep) as rPtr:
                self.keep_item = [item.rstrip() for item in rPtr.readlines()]

        self.viewset, item_view = [], []
        with open(self.path) as rPtr:
            idx = 0
            for line in rPtr:
                if (idx % 12) == 0:
                    if item_view:
                        self.viewset.append(item_view)
                    item_view = []
                if self.phase in ('train', 'val'):
                    line_infos = line.rstrip().split(' ')
                    if line_infos[0].split('/')[1] in self.keep_item:
                        item_view.append(
                            (os.path.join(base_dir, 'data', line_infos[0]),
                            int(line_infos[1])))
                else:
                    item_view.append(
                        os.path.join(base_dir, 'data', line.rstrip()))
                idx += 1
            if item_view:
                self.viewset.append(item_view)

    def __len__(self):
        return len(self.viewset)

    def __getitem__(self, index):
        images = []
        # file = None
        for item in self.viewset[index]:
            # file = item[0]
            if self.phase in ('trian', 'val'):
                image = Image.open(item[0]).convert('RGB')
            else:
                image = Image.open(item).convert('RGB')
            # image = image.resize((18, 18), Image.ANTIALIAS)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.cat([image.unsqueeze(0) for image in images], 0)
        if self.phase in ('train', 'val'):
            return images, self.viewset[index][0][1]#, file
        else:
            return images, self.viewset[index][0].split('/')[-2]


def MakeDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
