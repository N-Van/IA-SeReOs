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

def create_one_hot(mask, num_classes=3):
    one_hot_mask = torch.zeros([mask.shape[0], num_classes, mask.shape[1], mask.shape[2]], dtype=torch.float32)
    if mask.is_cuda:
        one_hot_mask = one_hot_mask.cuda()
    one_hot_mask = one_hot_mask.scatter(1, mask.long().data.unsqueeze(1), 1.0)
    #print(f"One-hot encoded mask unique values: {torch.unique(one_hot_mask)}")
    return one_hot_mask

def adjust_mask(mask, class_num):
    interval = int(256.0 / class_num)
    new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.longlong)
    for i in range(class_num):
        if i <= class_num - 2:
            new_mask[(mask >= i * interval) & (mask < (i + 1) * interval)] = i
        else:
            new_mask[i * interval <= mask] = i
    return new_mask

class adjustMask(object):
    def __init__(self, class_num=3):
        self.class_num = class_num

    def __call__(self, sample):
        sample['mask'] = adjust_mask(sample['mask'], self.class_num)
        return sample

class ToTensor(object):
    def __init__(self, if_multi_img=False):
        self.if_multi_img = if_multi_img

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # print("Before ToTensor - Mask unique values:", np.unique(mask))
        
        # Ensure image is in (C, H, W) format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # Convert (H, W) to (1, H, W) for single-channel images

        # Handle multi-image case
        if self.if_multi_img:
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=3)  # Convert (C, H, W) to (C, H, W, 1)
            image = image.transpose((0, 3, 1, 2))  # Convert (C, H, W, 1) to (C, 1, H, W)

        sample['image'] = torch.from_numpy(image)
        sample['mask'] = torch.from_numpy(mask)
        if 'weights' in sample:
            sample['weights'] = torch.from_numpy(sample['weights'])
        if 'ratio' in sample:
            sample['ratio'] = torch.from_numpy(sample['ratio'])
        # print("After ToTensor - Mask unique values:", np.unique(sample['mask']))
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
        if len(image.shape) == 2:  # If single channel, expand to 3 channels
            image = np.stack([image] * 3, axis=-1)
        sample['image'] = image
        return sample

class Augmentation(object):
    def __init__(self, output_size=None):
        self.output_size = output_size
        self.augmentation_pipeline = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(p=1),
                albu.VerticalFlip(p=1),
                albu.Compose([
                    albu.HorizontalFlip(p=1),
                    albu.VerticalFlip(p=1),
                ])
            ], p=0.75),
            albu.RandomRotate90(p=1),
        ])

        if self.output_size is not None:
            self.augmentation_pipeline.add_targets({'mask': 'mask'})
            self.augmentation_pipeline.transforms.append(albu.Resize(output_size, output_size))

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        
        # Ensure image and mask have the same height and width
        assert image.shape[1:] == mask.shape, f"Image and mask dimensions do not match. Image: {image.shape}, Mask: {mask.shape}"

        # Apply the augmentation pipeline
        augmented = self.augmentation_pipeline(image=image.transpose(1, 2, 0), mask=mask)
        sample['image'] = augmented['image'].transpose(2, 0, 1)  # Convert back to (C, H, W)
        sample['mask'] = augmented['mask']

        return sample