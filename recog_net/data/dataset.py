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


def json2bnj(json_file, img, b, n, s):
    w, h, _ = img.shape
    blur = np.zeros((w, h, 1))
    noise = np.zeros((w, h, 1))
    jpeg = np.zeros((w, h, 1))

    regions = json_file['regions']
    inten = json_file['intensity']

    for i in range(len(regions)):
        B = inten[i]["blur"]
        N = inten[i]["noise"]
        J = inten[i]["jpeg"]

        blur[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = b[B]
        noise[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = n[N]
        jpeg[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], :] = j[J]

    return blur, noise, jpeg


def make_dict():

    blur_sig = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
    noise_sig = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    jpeg_q = np.array([0, 15, 20, 25, 30, 35, 40, 50, 60, 80])
    intensity = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    invers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    return dict(zip(noise_sig, intensity)), dict(zip(blur_sig, intensity)), dict(zip(jpeg_q, invers))


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
        self.noise_dict, self.blur_dict, self.jpeg_dict = make_dict()

    def __getitem__(self, index):
        # size = self.size
        phase = self.phase
        img, json_file = self.X[index], self.seg[index]
        blur_label, noise_label, jpeg_label = json2bnj(json_file, img, self.blur_dict, self.noise_dict, self.jpeg_dict)
        filename = self.x_path[index].split('/')[-1]

        if phase == 'train':
            img = im2tensor(img)
            blur_label = im2tensor(blur_label)
            noise_label = im2tensor(noise_label)
            jpeg_label = im2tensor(jpeg_label)

            try:
                img, blur_label, noise_label, jpeg_label = random_crop(img, blur_label, noise_label, jpeg_label,
                                                                       size=self.size)
            except KeyError:
                print('{} data index'.format(str(index)))
                raise KeyError
            return img, (blur_label, noise_label, jpeg_label)

        else:
            img = im2tensor(img)
            blur_label = im2tensor(blur_label)
            noise_label = im2tensor(noise_label)
            jpeg_label = im2tensor(jpeg_label)

            return img, (blur_label, noise_label, jpeg_label), filename

    def __len__(self):
        return len(self.x_path)
