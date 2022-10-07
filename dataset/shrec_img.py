import os

import torch.utils.data as data
from PIL import Image

class ImageSet(data.Dataset):
    def __init__(self, phase='train', base_dir='./data',
                 retrieval=False,
                 transform=None):
        super(ImageSet, self).__init__()

        self.phase = phase
        self.retrieval = retrieval
        self.transform = transform

        assert phase in ('train', 'val', 'test')
        if phase == 'train':
            self.path = os.path.join(
                base_dir, 'img_train.txt')
        elif phase == 'val':
            self.path = os.path.join(
                base_dir, 'img_val.txt')
        else:
            self.path = os.path.join(base_dir, 'split',
                'shrec2019_Image_test.txt')
        with open(self.path) as rPtr:
            self.Sets = [os.path.join(base_dir, 'data',
                line.rstrip()) for line in rPtr.readlines()]

    def __len__(self):
        return len(self.Sets)

    def __getitem__(self, index):
        if self.phase in ('train', 'val'):
            infos = self.Sets[index].split(' ')
            image = Image.open(infos[0]).convert('RGB')
            label = int(infos[1])
        else:
            infos = self.Sets[index].rstrip()
            image = Image.open(infos).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.retrieval:
            if self.phase == 'test':
                return image, os.path.basename(infos).split('.')[0]
            return image, label, infos[0]
        return image, label


def MakeDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
