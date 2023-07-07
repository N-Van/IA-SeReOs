from __future__ import print_function, division
import os
import time
import torch
import numpy as np
import pandas as pd
import albumentations as albu
from PIL import Image
from torch.utils.data import Dataset

# this function only consider the situation of both mask and image are 2-D gray scale picture.
# That is the input is (H x W) not (H x W x C) even though the C is equal to 1.

def create_one_hot(mask, num_classes = 3):
    one_hot_mask = torch.zeros([mask.shape[0],
                                num_classes,
                                mask.shape[1],
                                mask.shape[2]],
                               dtype=torch.float32)
    if mask.is_cuda:
        one_hot_mask = one_hot_mask.cuda()
    one_hot_mask = one_hot_mask.scatter(1, mask.long().data.unsqueeze(1), 1.0)

    return one_hot_mask

def adjustMask(mask, class_num):

    interval = int(256.0 / class_num)

    # Color_Dict must be a numpy type
    # mask.shape must be a H x W x C
    # do not have channel dimensions
    if len(mask.shape) == 2:
        new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.longlong)
        for i in range(class_num):
            if i <= class_num - 2:
                new_mask[(mask >= i*interval) & (mask < (i+1) * interval)] = i
            else:
                new_mask[i*interval <= mask] = i
        return new_mask

class AdjustMask(object):
    def __init__(self, class_num = 3):
        self.class_num = class_num

    def __call__(self, sample):
        sample['mask'] = adjustMask(sample['mask'], self.class_num)
        return sample

class ToTensor(object):

    def __init__(self, if_multi_img=False):
        self.if_multi_img = if_multi_img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W

        if not self.if_multi_img:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = image.transpose((2, 0, 1))
        else:
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=3)

            image = image.transpose((0, 3, 1, 2))

        sample['image'] =  torch.from_numpy(image)
        sample['mask'] = torch.from_numpy(mask)

        if 'weights' in sample:
            sample['weights'] = torch.from_numpy(sample['weights'])
        if 'ratio' in sample:
            sample['ratio'] = torch.from_numpy(sample['ratio'])
        return sample

class Normalize(object):
    def __init__(self, max=255.0, min=0.0, tg_max=1.0, tg_min=0.0):
        self.max = max
        self.min = min
        self.tg_max = tg_max
        self.tg_min = tg_min

    def __call__(self, sample):
        image = sample['image'].astype('float32')
        image = self.tg_min + ((image - self.min)*(self.tg_max - self.tg_min)) / (self.max - self.min)
        sample['image'] = image
        return sample

class Augmentation(object):

    def __init__(self, output_size=256):
        self.aug = albu.Compose([
            albu.HorizontalFlip(),
            albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
            albu.ElasticTransform(alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.5),
            albu.ShiftScaleRotate(rotate_limit=180),
            albu.Resize(output_size, output_size, always_apply=True),
        ])
    def __call__(self,sample):
        augmented = self.aug(image=sample['image'], mask=sample['mask'])
        sample['image'] = augmented['image']
        sample['mask'] = augmented['mask']
        return sample