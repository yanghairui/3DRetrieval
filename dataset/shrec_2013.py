""" SHREC 2013 - Large Scale Sketch-Based 3D Shape Retrieval
DataSet: http://orca.st.usm.edu/~bli/sharp/sharp/contest/2013/SBR/data.html
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image
# from render import RENDER


class shrec_model(data.Dataset):
    def __init__(self, path):
        super(shrec_model, self).__init__()
        self.root_path = path
        self.model_class_file = os.path.join(self.root_path, 'SHREC2013_Sketch_Evaluation', 'SHREC13_SBR_Model.cla')
        self.model_path = os.path.join(self.root_path, 'SHREC13_SBR_TARGET_MODELS', 'models')
        self.model_image_path = os.path.join(self.root_path, 'SHREC13_SBR_TARGET_MODELS', 'views')
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
                return class_idx, os.path.join(self.model_path, 'm' + self.model_dict[class_name][index - acc] + '.off')
            acc += len(self.model_dict[class_name])

    def views(self, off_file, num_views=12, save=False, model_name=None, size=224):
        images = RENDER.images(off_file, num_views)
        if save:
            self._views_save(images, model_name, size)
        return images

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
        if self.phase == 'train':
            self.set = self._read_file(os.path.join(path, 'SHREC13_SBR_TRAINING_SKETCHES'))
        elif self.phase == 'test':
            self.set = self._read_file(os.path.join(path, 'SHREC13_SBR_TESTING_SKETCHES'))
        else:
            self.set = self._read_file(os.path.join(path, 'SHREC13_SBR_TRAINING_SKETCHES'))
            self.set.extend(self._read_file(os.path.join(path, 'SHREC13_SBR_TESTING_SKETCHES')))

    def _read_file(self, path):
        _set = []
        for item in glob.glob(os.path.join(path, '*', '*', '*.png')):
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
    def __init__(self, path, phase='train', transform=None):
        super(shrec_view, self).__init__()
        assert phase in ('train', 'val', 'all')
        self.transform = transform
        self.phase = phase
        self.path = os.path.join(path, 'SHREC13_SBR_TARGET_MODELS')
        if self.phase == 'train':
            self.set = self._read_file(os.path.join(self.path, 'train.txt'))
        elif self.phase == 'val':
            self.set = self._read_file(os.path.join(self.path, 'val.txt'))
        else:
            self.set = self._read_file(os.path.join(self.path, 'train.txt'))
            self.set.extend(self._read_file(os.path.join(self.path, 'val.txt')))

    def _read_file(self, file_path):
        _set = []
        with open(file_path) as ptr:
            for line in ptr:
                line_infos = line.rstrip().split(' ')
                for sample in line_infos[2:]:
                    sample_path = os.path.join(self.path, 'views', 'm' + sample)
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


class shrec_view_show(data.Dataset):
    def __init__(self, path, size=12):
        super(shrec_view_show, self).__init__()
        self.size = size
        self.path = os.path.join(path, 'SHREC13_SBR_TARGET_MODELS')
        self.set = self._read_file(os.path.join(self.path, 'train.txt'))
        self.set.extend(self._read_file(os.path.join(self.path, 'val.txt')))

    def _read_file(self, file_path):
        _set = []
        with open(file_path) as ptr:
            for line in ptr:
                line_infos = line.rstrip().split(' ')
                for sample in line_infos[2:]:
                    sample_path = os.path.join(self.path, 'views', 'm' + sample)
                    _set.append([line_infos[0], line_infos[1], sample_path])
        return _set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        model_name, class_idx, file_path = self.set[index]
        for item in glob.glob(os.path.join(file_path, '*.jpg')):
            image = np.array(Image.open(item).convert('RGB').resize((self.size, self.size)))
            break
        return image


if __name__ == '__main__':
    path = "C:/Users/HEDGEHOG/Desktop/3DRetrieval/data/SHREC_2013"
    model = shrec_model(path=path)
    # X, y = [], []
    # total_nums = 0
    # for model_name in model.model_dict.keys():
    #     print(model_name, model.get_model_nums(model_name), model.get_model_index(model_name))
    #     X.append(model.get_model_index(model_name))
    #     y.append(model.get_model_nums(model_name))
    #     total_nums += model.get_model_nums(model_name)
    # print(len(model), total_nums)
    #
    # plt.scatter(X, y)
    # plt.savefig('dis13.png')
    # plt.show()

    # for sample_idx in range(len(model)):
    #     class_idx, off_file = model[sample_idx]
    #     # images = model.views(off_file, save=True, model_name=os.path.basename(off_file).split('.')[0])

    # OUTPUT_DIR = os.path.join(path, 'SHREC13_SBR_TARGET_MODELS')
    # with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as trainPtr:
    #     with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as valPtr:
    #         for key in model.model_dict.keys():
    #             models = np.array(model.model_dict[key])
    #             np.random.shuffle(models)
    #             models = list(models)
    #             _size = len(models)
    #             class_idx = str(model.get_model_index(key))
    #             train, val = [key, class_idx], [key, class_idx]
    #             if _size <= 10:
    #                 val.extend(models[0:1])
    #                 train.extend(models[1:])
    #             elif 10 < _size <= 15:
    #                 val.extend(models[0:2])
    #                 train.extend(models[2:])
    #             elif 15 < _size <= 20:
    #                 val.extend(models[0:3])
    #                 train.extend(models[3:])
    #             elif 20 < _size <= 30:
    #                 val.extend(models[0:5])
    #                 train.extend(models[5:])
    #             elif 30 < _size <= 40:
    #                 val.extend(models[0:7])
    #                 train.extend(models[7:])
    #             elif 40 < _size <= 50:
    #                 val.extend(models[0:10])
    #                 train.extend(models[10:])
    #             elif 50 < _size <= 80:
    #                 val.extend(models[0:14])
    #                 train.extend(models[14:])
    #             elif 80 < _size <= 100:
    #                 val.extend(models[0:20])
    #                 train.extend(models[20:])
    #             elif 100 < _size <= 150:
    #                 val.extend(models[0:35])
    #                 train.extend(models[35:])
    #             else:
    #                 val.extend(models[0:79])
    #                 train.extend(models[79:])
    #
    #             trainPtr.write(' '.join(train) + '\n')
    #             valPtr.write(' '.join(val) + '\n')

    import torchvision.transforms as transform
    view_sets = shrec_view(path, 'all', transform.ToTensor())
    print(len(view_sets))
    images, class_idx, model_name = view_sets[0]
    print(images.size(), class_idx, model_name)

    image_sets = shrec_image(path, 'all', transform.ToTensor())
    print(len(image_sets))
    image, class_idx, model_name = image_sets[-1]
    print(image.size(), class_idx, model_name)
