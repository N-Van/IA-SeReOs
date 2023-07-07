from __future__ import print_function, division
import re
import h5py
import numpy as np
import pandas as pd
import utils.dataprocess as dp
from torchvision import transforms
from torch.utils.data import Dataset


def alpha_to_int(text):
    clean_text = int(text) if text.isdigit() else text
    return clean_text

def alpha_to_float(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [alpha_to_int(c) for c in re.split(r'(\d+)', text)]

def natural_keys_float(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [alpha_to_float(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

def load_patches(patches):

    if isinstance(patches, str):
        return np.array(pd.read_csv(patches, header=0)).tolist()
    else:
        return patches

class HDF52D(Dataset):

    # dataset for segmentation used
    def __init__(self, data_path, train_patches,val_patches, train_transform=None, val_transform=None, train_idx = None):

        self.data_path = data_path

        self.patches = {'train': load_patches(train_patches),
                        'val': load_patches(val_patches)}

        self.transforms = {'train': train_transform,
                           'val': val_transform}

        self.train_idx = load_patches(train_idx)

        self.mode = 'train'

    def __getitem__(self, idx):

        [name, top, left, h, w] = self.patches[self.mode][idx]

        with h5py.File(self.data_path,'r') as f:
            image = f[name]['data'][top:top + h, left:left + w ]
            mask = f[name]['label'][top:top + h, left:left + w]
            sample = {'image': image, 'mask': mask}

        if self.transforms[self.mode] is not None:
            sample = self.transforms[self.mode](sample)
        if self.train_idx is not None and self.mode == 'train':
            sample['index'] = self.train_idx[idx]
        return sample


    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def __len__(self):
        return len(self.patches[self.mode])

if __name__ == '__main__':
    data_path = '/cvdata/yungchen/rdn_revised/data/dataset.hdf5'
    train_patches = '/cvdata/yungchen/rdn_revised/data/patches.csv'
    val_patches = '/cvdata/yungchen/rdn_revised/data/val.csv'
    ratios = '/cvdata/yungchen/rdn_revised/data/ratios.csv'

    transforms = transforms.Compose([dp.Augmentation(output_size=256),
                                     dp.AdjustMask(class_num=3),
                                     dp.Normalize(max=255, min=0)])

    data_set = HDF52D(data_path,train_patches,val_patches,train_transform=transforms, train_idx=ratios)
    sample = data_set[1000]
    mask = sample['mask']

    print(np.sum(mask == 0))
    print(np.sum(mask == 1))
    print(np.sum(mask == 2))
    ...
