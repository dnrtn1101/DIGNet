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


def random_crop(img, blur, noise, jpeg, size):
    h, w = img.shape[1:]
    x = random.randint(0, h - size)
    y = random.randint(0, w - size)

    crop_img = img[:, x:x + size, y:y + size]
    crop_blur = blur[:, x:x + size, y:y + size]
    crop_noise = noise[:, x:x + size, y:y + size]
    crop_jpeg = jpeg[:, x:x + size, y:y + size]
    return crop_img, crop_blur, crop_noise, crop_jpeg


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


def json2bnj(json_file, img, d, f, s):
    w, h, _ = img.shape
    defocus = np.zeros((w, h, 1))
    fnoise = np.zeros((w, h, 1))
    snow = np.zeros((w, h, 1))

    regions = json_file['regions']
    inten = json_file['intensity']

    for i in range(len(regions)):
        D = inten[i]["defocus"]
        F = inten[i]["fnoise"]
        if inten[i]["snow"] == 0:
            snow[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = 0
        else:
            S = sum(inten[i]["snow"])
            snow[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = s[S]

        defocus[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = d[D]
        fnoise[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = f[F]

    return defocus, fnoise, snow


def make_dict():

    defocus_sig = np.array([0, 1., 2., 3., 4., 5., 6., 7., 8., 10.])
    fnoise_sig = np.array([0, 500., 250., 150., 100., 80., 60., 40., 25., 15.])
    snow_sig = np.array([0, sum([0.1, 0.8, 4, 0.97]),
                         sum([0.1, 0.75, 4, 0.95]),
                         sum([0.15, 0.75, 4, 0.93]),
                         sum([0.2, 0.75, 5, 0.91]),
                         sum([0.2, 0.7, 5, 0.89]),
                         sum([0.25, 0.8, 5, 0.87]),
                         sum([0.25, 0.7, 5, 0.85]),
                         sum([0.3, 0.7, 6, 0.83]),
                         sum([0.35, 0.7, 6, 0.81])])
    intensity = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return dict(zip(defocus_sig, intensity)), dict(zip(fnoise_sig, intensity)), dict(zip(snow_sig, intensity))


class Dataset(data.Dataset):
    def __init__(self, phase, data_root, size, level):
        super(Dataset, self).__init__()
        self.level = level
        self.phase = phase
        self.data_root = data_root
        self.x_path, self.seg_path = list(), list()
        self.x_path += sorted(glob.glob(os.path.join(self.data_root, self.phase, '*.png')))
        self.seg_path += sorted(
            glob.glob(os.path.join(self.data_root, '{}_label'.format(self.phase), '*.json')))
        self.size = size

        self.X, self.seg = list(), list()
        for x_path, seg_path in zip(self.x_path, self.seg_path):
            self.X += [io.imread(x_path)]
            with open(seg_path, 'r') as f:
                self.seg += [json.load(f)]

        # for making mask
        self.defocus_dict, self.fnoise_dict, self.snow_dict = make_dict()

    def __getitem__(self, index):
        phase = self.phase
        img, json_file = self.X[index], self.seg[index]
        defocus_label, fnoise_label, snow_label = json2bnj(json_file, img, self.defocus_dict, self.fnoise_dict,
                                                       self.snow_dict)
        filename = self.x_path[index].split('/')[-1]

        if phase == 'train':
            img = im2tensor(img)
            defocus_label = im2tensor(defocus_label)
            fnoise_label = im2tensor(fnoise_label)
            snow_label = im2tensor(snow_label)

            try:
                img, defocus_label, fnoise_label, snow_label = random_crop(img, defocus_label, fnoise_label, snow_label,
                                                                       size=self.size)
            except KeyError:
                print('{} data index'.format(str(index)))
                raise KeyError
            return img, (defocus_label, fnoise_label, snow_label)

        else:
            img = im2tensor(img)
            defocus_label = im2tensor(defocus_label)
            fnoise_label = im2tensor(fnoise_label)
            snow_label = im2tensor(snow_label)

            return img, (defocus_label, fnoise_label, snow_label), filename

    def __len__(self):
        return len(self.x_path)
