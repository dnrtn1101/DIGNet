import os
import glob
import random
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import skimage.io as io
import pdb
import json


#
def random_crop(data, target, size):
    h, w = data.shape[1:]
    x = random.randint(0, h - size)
    y = random.randint(0, w - size)

    crop_data = data[:, x:x + size, y:y + size]
    crop_target = target[:, x:x + size, y:y + size]

    return crop_data, crop_target


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


class Dataset(data.Dataset):
    def __init__(self, phase, data_root, size, level):
        super(Dataset, self).__init__()

        self.level = level
        self.phase = phase
        self.size = size
        self.data_root = data_root

        self.x_path = sorted(glob.glob(os.path.join(self.data_root, self.phase, self.level, '*.png')))
        if self.phase == 'train':
            self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png'))) * 3
        else:
            self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png')))
        self.X, self.Y = list(), list()

        for x_path, y_path in zip(self.x_path, self.y_path):
            self.X += [io.imread(x_path)]
            self.Y += [io.imread(y_path)]

    def __getitem__(self, index):
        size = self.size
        phase = self.phase
        img, target = self.X[index], self.Y[index]
        filename = self.x_path[index].split('/')[-1]

        if phase == 'train':
            img = im2tensor(img)
            target = im2tensor(target)
            try:
                img, target = random_crop(img, target, size=size)
            except KeyError:
                print('{} data index'.format(str(index)))
                raise KeyError
            return img, target
        else:
            img = im2tensor(img)
            target = im2tensor(target)

            return img, target, filename

    def __len__(self):
        return len(self.x_path)
########################################################################################################################
########################################################################################################################
# def random_crop(data, target, mask, size):
#     h, w = data.shape[1:]
#     x = random.randint(0, h - size)
#     y = random.randint(0, w - size)
#
#     crop_data = data[:, x:x + size, y:y + size]
#     crop_target = target[:, x:x + size, y:y + size]
#     crop_mask = mask[:, x:x + size, y:y + size]
#     return crop_data, crop_target, crop_mask
#
# # def random_crop(data, target, size):
# #     h, w = data.shape[1:]
# #     x = random.randint(0, h - size)
# #     y = random.randint(0, w - size)
# #
# #     crop_data = data[:, x:x + size, y:y + size]
# #     crop_target = target[:, x:x + size, y:y + size]
# #
# #     return crop_data, crop_target
#
# def im2tensor(im):
#     np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
#     tensor = torch.from_numpy(np_t).float()
#     return tensor
#
# def json2inten(json_file, img, b, n, j):
#     mask = np.zeros(img.shape)
#     regions = json_file['regions']
#     inten = json_file['intensity']
#
#     for i in range(len(regions)):
#         # B = inten[i]["blur"]
#         # N = inten[i]["noise"]
#         # J = inten[i]["jpeg"]
#         B = inten[i]["defocus"]
#         N = inten[i]["fnoise"]
#         if inten[i]["snow"] == 0:
#             mask[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], 2] = 0
#         else:
#             S = sum(inten[i]["snow"])
#             mask[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], 2] = s[S]
#         mask[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], 0] = b[B]
#         mask[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], 1] = n[N]
#         mask[regions[i][0]:regions[i][1], regions[i][2]:regions[i][3], 2] = j[J]
#
#     return mask
#
# def make_dict():
#     blur_sig = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
#     noise_sig = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
#     jpeg_q = np.array([0, 15, 20, 25, 30, 35, 40, 50, 60, 80])
#
#     defocus_sig = np.array([0, 1., 2., 3., 4., 5., 6., 7., 8., 10.])
#     snow_sig = np.array([0, sum([0.1, 0.8, 4, 0.97]),
#                          sum([0.1, 0.75, 4, 0.95]),
#                          sum([0.15, 0.75, 4, 0.93]),
#                          sum([0.2, 0.75, 5, 0.91]),
#                          sum([0.2, 0.7, 5, 0.89]),
#                          sum([0.25, 0.8, 5, 0.87]),
#                          sum([0.25, 0.7, 5, 0.85]),
#                          sum([0.3, 0.7, 6, 0.83]),
#                          sum([0.35, 0.7, 6, 0.81])])
#     fnoise_sig = np.array([0, 500., 250., 150., 100., 80., 60., 40., 25., 15.])
#     intensity = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     invers = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
#
#     # return dict(zip(noise_sig, intensity)), dict(zip(blur_sig, intensity)), dict(zip(jpeg_q, invers))
#     return dict(zip(defocus_sig, intensity)), dict(zip(fnoise_sig, intensity)), dict(zip(snow_sig, intensity))
#
# class Dataset(data.Dataset):
#     def __init__(self, phase, data_root, size, level):
#         super(Dataset, self).__init__()
#
#         self.level = level
#         self.phase = phase
#         self.size = size
#         self.data_root = data_root
#         # if self.data_root == 'sidd':
#         #     if self.phase == 'train':
#         #         self.x_path = sorted(glob.glob('./sidd_train/input_crop/*'))
#         #         self.y_path = sorted(glob.glob('./sidd_train/target_crop/*'))
#         #         self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_sidd/train_crop/*'))
#         #     else:
#         #         self.x_path = sorted(glob.glob('./sidd_val/input_img/*'))
#         #         self.y_path = sorted(glob.glob('./sidd_val/target_img/*'))
#         #         self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_sidd/val/*'))
#         #
#         # elif self.data_root == 'gopro':
#         #     if self.phase == 'train':
#         #         self.x_path = sorted(glob.glob('./gopro/train/blur/*'))
#         #         self.y_path = sorted(glob.glob('./gopro/train/sharp/*'))
#         #         self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_gopro/train/*'))
#         #     else:
#         #         self.x_path = sorted(glob.glob('./gopro/test/blur/*'))
#         #         self.y_path = sorted(glob.glob('./gopro/test/sharp/*'))
#         #         self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_gopro/val/*'))
#         #
#         # elif self.data_root == 'goprogamma':
#         #     if self.phase == 'train':
#         #         self.x_path = sorted(glob.glob('./gopro/train/blur_gamma/*'))
#         #         self.y_path = sorted(glob.glob('./gopro/train/sharp/*'))
#         #         # self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_goprogamma/train/*'))
#         #     else:
#         #         self.x_path = sorted(glob.glob('./gopro/test/blur_gamma/*'))
#         #         self.y_path = sorted(glob.glob('./gopro/test/sharp/*'))
#         #         # self.seg_path = sorted(glob.glob('./subnet/output/unet_aspp_goprogamma/val/*'))
#         # else:
#
#         self.x_path = sorted(glob.glob(os.path.join(self.data_root, self.phase, self.level, '*.png')))
#         if self.phase == 'train':
#             self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png'))) * 3
#         else:
#             self.y_path = sorted(glob.glob(os.path.join(self.data_root[:-2], self.phase, '*target', '*.png')))
#         # self.seg_path = sorted(
#         #     glob.glob(os.path.join(self.data_root, self.phase, '{}_label'.format(self.level), '*.json')))
#         self.seg_path = sorted(
#             glob.glob(os.path.join('./subnet/output/unet_aspp_dataset4/', self.phase, self.level, '*png')))
#
#         self.X, self.Y, self.seg = list(), list(), list()
#         # self.X, self.Y = list(), list()
#         for x_path, y_path, seg_path in zip(self.x_path, self.y_path, self.seg_path):
#             # for x_path, y_path in zip(self.x_path, self.y_path):
#             self.X += [io.imread(x_path)]
#             self.Y += [io.imread(y_path)]
#             self.seg += [io.imread(seg_path)]
#             # with open(seg_path, 'r') as f:
#             #     self.seg += [json.load(f)]
#
#         # mask 를 만들기 위한..
#         # self.noise_dict, self.blur_dict, self.jpeg_dict = make_dict()
#         # self.defocus_dict, self.fnoise_dict, self.snow_dict = make_dict()
#
#     def __getitem__(self, index):
#         size = self.size
#         phase = self.phase
#         img, target, mask = self.X[index], self.Y[index], self.seg[index]
#         # img, target = self.X[index], self.Y[index]
#         ## add seg mask
#         # with open(self.seg_path[index], 'r') as f:
#         #     json_file = json.load(f)
#
#         # mask = json2inten(mask, img, self.defocus_dict, self.fnoise_dict, self.snow_dict)
#         # mask = json2inten(mask, img, self.blur_dict, self.noise_dict, self.jpeg_dict)
#
#         filename = self.x_path[index].split('/')[-1]
#
#         if phase == 'train':
#             img = im2tensor(img)
#             target = im2tensor(target)
#             mask = im2tensor(mask)
#             #
#             try:
#                 img, target, mask = random_crop(img, target, mask, size=size)
#             except KeyError:
#                 print('{} data index'.format(str(index)))
#                 raise KeyError
#             return img, target, mask
#             # try:
#             #     img, target = random_crop(img, target, size=size)
#             # except KeyError:
#             #     print('{} data index'.format(str(index)))
#             #     raise KeyError
#             # return img, target
#         else:
#             img = im2tensor(img)
#             target = im2tensor(target)
#             mask = im2tensor(mask)
#
#             return img, target, mask, filename
#             # return img, target, filename
#
#     def __len__(self):
#         return len(self.x_path)
#
