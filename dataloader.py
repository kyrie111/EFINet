import os
import math
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import random

random.seed(1143)


def data_aug(img1, img2):
    a = random.random()
    b = math.floor(random.random() * 4)
    if a >= 0.5:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if b == 1:
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    elif b == 2:
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    elif b == 3:
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    return img1, img2


class DataSet_loader(data.Dataset):

    def __init__(self, transform1, is_gth_train, path=None, flag='train'):
        self.flag = flag
        self.transform1 = transform1
        self.dark_path, self.gt_path = path
        self.dark_data_list = os.listdir(self.dark_path)
        self.dark_data_list.sort(key=lambda x: int(x[:-4]))
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        if is_gth_train:
            self.dark_data_list = self.dark_data_list + self.gt_data_list
        self.length = len(self.dark_data_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dark_name = self.dark_data_list[idx][:-4]
        gth_name = dark_name
        gt_image = Image.open(self.gt_path + gth_name + '.jpg')
        dark_image = Image.open(self.dark_path + dark_name + '.jpg')
        # 数据增强
        if self.flag == 'train':
            dark_image, gt_image = data_aug(dark_image, gt_image)
            
        dark_image = np.asarray(dark_image)
        gt_image = np.asarray(gt_image)

        if self.transform1:
            dark_image = self.transform1(dark_image)
            gt_image = self.transform1(gt_image)

        return dark_name, dark_image, gt_image
