import os
import glob
import random
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import skimage.io as io
import json


def random_crop(data, target, mask, size):
    h, w = data.shape[1:]
    x = random.randint(0, h-size)
    y = random.randint(0, w-size)

    crop_data = data[:, x:x + size, y:y + size]
    crop_target = target[:, x:x + size, y:y + size]
    crop_mask = mask[:, x:x + size, y:y + size]
    return crop_data, crop_target, crop_mask


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


class Dataset(data.Dataset):
    def __init__(self, phase, data_root, size):
        super(Dataset, self).__init__()

        self.phase = phase
        self.size = size
        self.data_root = data_root
        self.mask_root = mask_root

        self.x_path = sorted(glob.glob(os.path.join(self.data_root, self.phase, '*.png')))
        if self.phase == 'train':
            self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png'))) * 3
        else:
            self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png')))
        self.seg_path = sorted(
            glob.glob(os.path.join(self.mask_root[:-2], self.phase, '*png')))
        self.X, self.Y, self.seg = list(), list(), list()

        for x_path, y_path, seg_path in zip(self.x_path, self.y_path, self.seg_path):
            self.X += [io.imread(x_path)]
            self.Y += [io.imread(y_path)]
            self.seg += [io.imread(seg_path)]

    def __getitem__(self, index):
        size = self.size
        phase = self.phase
        img, target, mask = self.X[index], self.Y[index], self.seg[index]
        filename = self.x_path[index].split('/')[-1]

        if phase == 'train':
            img = im2tensor(img)
            target = im2tensor(target)
            try:
                img, target, mask = random_crop(img, target, mask,  size=size)
            except KeyError:
                print('{} data index'.format(str(index)))
                raise KeyError
            return img, target, mask
        else:
            img = im2tensor(img)
            target = im2tensor(target)
            mask = im2tensor(mask)
            return img, target, mask, filename

    def __len__(self):
        return len(self.x_path)