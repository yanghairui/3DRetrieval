import os.path as osp
import os
import glob

import torch.utils.data as data

from PIL import Image

# from render import RENDER


class shrec_3DSketch(data.Dataset):
    def __init__(self, path, phase='train', transform=None):
        super(shrec_3DSketch, self).__init__()
        assert phase in ('train', 'test')

        self.transform = transform
        self.data_dir = osp.join(path, 'SHREC16_3DSBR_Benchmark')
        if phase == 'train':
            self.cla = osp.join(self.data_dir, 'Training_Testing_Target_Datasets_Classficiation_FIles', 'Kinect300_Query_Train.cla')
        else:
            self.cla = osp.join(self.data_dir, 'Training_Testing_Target_Datasets_Classficiation_FIles', 'Kinect300_Query_Test.cla')
        map_index_file = osp.join(self.data_dir, 'Training_Testing_Target_Datasets_Classficiation_FIles', 'map_index.txt')
        self.map_indx = {}
        with open(map_index_file) as rPtr:
            for line in rPtr.readlines():
                key, indx, map_indx = line.strip().split(' ')
                self.map_indx[key] = (indx, map_indx)

        self._reader()

    def _reader(self):
        self.set = []
        key = ''
        with open(self.cla) as rPtr:
            for line in rPtr.readlines():
                infos = line.strip().split(' ')
                if len(infos) == 0 or len(infos) == 2:
                    continue
                elif len(infos) == 3:
                    key, _, _ = infos
                elif len(infos) == 1 and infos[0] != '':
                    if key in self.map_indx.keys():
                        self.set.append((infos[0], self.map_indx[key][0], self.map_indx[key][1]))

    def __getitem__(self, index):
        _id, indx, map_indx = self.set[index]
        try:
            image_file = osp.join(self.data_dir, 'Kinect300_2DView', str(_id) + '.png')
            image = Image.open(image_file).convert('RGB')
        except:
            image_file = osp.join(self.data_dir, 'Kinect300_2DView', str(_id) + '.jpg')
            image = Image.open(image_file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(indx), int(map_indx)

    def __len__(self):
        return len(self.set)


if __name__ == '__main__':
    src_path = 'C:/Users/HEDGEHOG/Desktop/3DRetrieval/data/SHREC_2016'
    sketch = shrec_3DSketch(src_path, phase='test')
    image, indx, map_indx = sketch[0]
    print(image.size, indx, map_indx)
