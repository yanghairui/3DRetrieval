""" SHREC’14 Track: Extended Large Scale Sketch-Based 3D Shape Retrieval
DataSet: http://orca.st.usm.edu/~bli/sharp/sharp/contest/2014/SBR/data.html
"""

import glob
import os

import torch
import torch.utils.data as data

from PIL import Image
# from render import RENDER


class shrec_model(data.Dataset):
    def __init__(self, path):
        super(shrec_model, self).__init__()
        self.root_path = path
        self.model_class_file = os.path.join(self.root_path, 'SHREC14_Sketch_Evaluation', 'SHREC14_SBR_Model.cla')
        self.model_path = os.path.join(self.root_path, 'SHREC14LSSTB_TARGET_MODELS')
        self.model_image_path = os.path.join(self.root_path, 'views')
        if not os.path.isdir(self.model_image_path):
            os.mkdir(self.model_image_path)
        self._read_file()

    def _read_file(self):
        self.model_dict = {}
        with open(self.model_class_file) as rPtr:
            for line in rPtr:
                line_infos = line.strip().split(' ')
                if len(line_infos) == 2 or (len(line_infos) == 1 and line_infos[0] == ''):
                    if len(line_infos) == 2:
                        try:
                            self.class_nums, self.total_nums = int(line_infos[0].strip()), int(line_infos[1].strip())
                        except:
                            pass
                    continue
                elif len(line_infos) >= 3:
                    if line_infos[0].strip() not in self.model_dict:
                        cur_key = line_infos[0].strip()
                        self.model_dict[cur_key] = []
                else:
                    self.model_dict[cur_key].append(line_infos[0].strip())

    def get_model_nums(self, model_name):
        return len(self.model_dict[model_name])

    def get_model_index(self, model_name):
        for idx, name in enumerate(self.model_dict.keys()):
            if name == model_name:
                return idx
        return -1

    def __len__(self):
        return self.total_nums

    def __getitem__(self, index):
        acc = 0
        for class_idx, class_name in enumerate(self.model_dict.keys()):
            if acc <= index < (len(self.model_dict[class_name]) + acc):
                return class_idx, os.path.join(self.model_path, 'M' + self.model_dict[class_name][index - acc] + '.off')
            acc += len(self.model_dict[class_name])

    def views(self, off_file, num_views=12, save=False, model_name=None, size=224):
        try:
            images = RENDER.images(off_file, num_views)
            if save:
                self._views_save(images, model_name, size)
            return images
        except:
            print(off_file)

    def _views_save(self, images, model_name, size):
        if isinstance(size, int):
            size = (size, size)
        DIR = os.path.join(self.model_image_path, model_name)
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
        for image_idx, image in enumerate(images):
            image = image.resize(size, Image.BICUBIC)
            image.save(os.path.join(DIR, '{}.jpg'.format(image_idx)))


class shrec_image(data.Dataset):
    def __init__(self, path, phase='train', transform=None, retrieval=False):
        super(shrec_image, self).__init__()
        assert phase in ('train', 'test', 'all')
        self.retrieval = retrieval
        self.transform = transform
        self.phase = phase
        self.model = shrec_model(path)
        if self.phase in ('train', 'test'):
            self.set = self._read_file(os.path.join(path, 'SHREC14LSSTB_SKETCHES'), self.phase)
        else:
            self.set = self._read_file(os.path.join(path, 'SHREC14LSSTB_SKETCHES'), 'train')
            self.set.extend(self._read_file(os.path.join(path, 'SHREC14LSSTB_SKETCHES'), 'test'))

    def _read_file(self, path, phase):
        _set = []
        for item in glob.glob(os.path.join(path, 'SHREC14LSSTB_SKETCHES', '*', phase, '*.png')):
            # print(item)
            model_name = item.split('/')[-3]
            _set.append([model_name, self.model.get_model_index(model_name), item])
        return _set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        model_name, class_idx, file_path = self.set[index]
        instance_name = os.path.basename(file_path).split('.')[0]
        image = Image.open(file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.retrieval:
            return image, int(class_idx), instance_name
        return image, int(class_idx)


class shrec_view(data.Dataset):
    def __init__(self, path, phase='all', transform=None):
        super(shrec_view, self).__init__()
        assert phase in ('train', 'val', 'all')
        self.transform = transform
        self.phase = phase
        self.path = path
        self.set = self._read_file(os.path.join(self.path, phase + '.txt'))

    def _read_file(self, file_path):
        _set = []
        with open(file_path) as ptr:
            for line in ptr:
                line_infos = line.rstrip().split(' ')
                for sample in line_infos[2:]:
                    sample_path = os.path.join(self.path, 'views', 'M' + sample)
                    _set.append([line_infos[0], line_infos[1], sample_path])
        return _set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        model_name, class_idx, file_path = self.set[index]
        images = []
        for item in glob.glob(os.path.join(file_path, '*.jpg')):
            instance_name = item.split('/')[-2]
            image = Image.open(item).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.cat([image.unsqueeze(0) for image in images], 0)

        if self.phase == 'all':
            return images, int(class_idx)#, instance_name
        return images, int(class_idx)


if __name__ == '__main__':
    path = 'C:/Users/HEDGEHOG/Desktop/3DRetrieval/data/SHREC_2014'
    MODEL = shrec_model(path)
    # for sample_idx in range(len(MODEL)):
    #     # print('{}/{}'.format(sample_idx, len(MODEL)))
    #     class_idx, off_file = MODEL[sample_idx]
    #     images = MODEL.views(off_file, save=True, model_name=os.path.basename(off_file).split('.')[0])

    # class_idx, off_file = MODEL[2991]
    # print(class_idx, off_file)
    # images = MODEL.views(off_file, save=True, model_name=os.path.basename(off_file).split('.')[0])

    # with open('all.txt', 'w') as wptr:
    #     for key in MODEL.model_dict.keys():
    #         infos = key + ' ' + str(MODEL.get_model_index(key)) + ' ' + ' '.join(MODEL.model_dict[key])
    #         wptr.write(infos + '\n')

    image_set = shrec_image(path)
    print(type(image_set[0]))
