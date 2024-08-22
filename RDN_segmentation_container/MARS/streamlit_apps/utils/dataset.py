from __future__ import print_function, division
import re
import h5py
import os
import numpy as np
import pandas as pd
import utils.dataprocess as dp
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


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






# Function to get the filename prefix
def get_filename_prefix(directory):
    """
    Get the common prefix of the filenames in the directory.
    Handles cases like 'os_long_0019.tif' and returns 'os_long'.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    if not files:
        raise ValueError(f"No .tif files found in the directory: {directory}")
    
    # Split the first file on the underscore and join all but the last part
    prefix = '_'.join(files[0].split('_')[:-1])
    return prefix


# Function to get the number of digits in the file names
def get_num_digits(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    if not files:
        raise ValueError(f"No .tif files found in the directory: {directory}")
    num_digits = len(files[0].split('_')[-1].split('.')[0])
    return num_digits

# Function to get the neighbor paths
def get_neighbor_paths(image_index, directory, n_channels, step):
    # Get the filename prefix
    filename_prefix = get_filename_prefix(directory)
    
    # Get the start and end index of the images in the directory
    start_index, end_index = get_image_indices(directory, filename_prefix)
    
    # Get the number of digits
    num_digits = get_num_digits(directory)
    
    neighbors = []
    
    # Calculate neighbor indices
    for i in range(-(n_channels // 2), (n_channels // 2) + 1):
        neighbor_index = image_index + i * step
        
        # Handle borders by repeating the border image
        if neighbor_index < start_index:
            neighbor_index = start_index
        elif neighbor_index > end_index:
            neighbor_index = end_index
        
        neighbors.append(neighbor_index)
    
    # Generate the neighbor paths using the correct number of digits
    neighbor_paths = [
        os.path.join(directory, f"{filename_prefix}_{n:0{num_digits}d}.tif").replace("\\", "/") 
        for n in neighbors
    ]
    
    return neighbor_paths

# Helper function to get the start and end indices in the directory
def get_image_indices(directory, filename_prefix):
    """
    Scan the directory to identify the start and end indices of the image files.
    """
    file_names = [f for f in os.listdir(directory) if f.startswith(filename_prefix) and f.endswith('.tif')]
    
    # Filter out files that do not have a numeric suffix
    indices = sorted([int(f.split('_')[-1].split('.')[0]) for f in file_names if f.split('_')[-1].split('.')[0].isdigit()])
    
    if not indices:
        raise ValueError(f"No valid image files found in directory '{directory}' with prefix '{filename_prefix}'")
    
    # Return the smallest and largest indices
    return indices[0], indices[-1]


def load_2_5D_image(image_index, directory, n_channels, step):
    if n_channels == 1:
        # Get the number of digits
        num_digits = get_num_digits(directory)
        
        # Load the single image directly
        image_path = os.path.join(directory, f"os_long_mini_{image_index:0{num_digits}d}.tif").replace("\\", "/")
        image = Image.open(image_path).convert('L')
        return np.expand_dims(np.array(image), axis=0)  # Return (1, H, W) image

    else:
        # Logic for multi-channel (2.5D) images
        neighbor_paths = get_neighbor_paths(image_index, directory, n_channels, step)
        images = [Image.open(p).convert('L') for p in neighbor_paths if p is not None]
        image_stack = np.stack([np.array(img) for img in images], axis=0)  # Stack along the first dimension (C, H, W)
        return image_stack

class HDF52D(Dataset):
    def __init__(self, data_path, train_patches, val_patches, image_dir, n_channels, step, train_transform=None, val_transform=None, train_idx=None):
        self.data_path = data_path
        self.image_dir = image_dir  # Directory where the .tif images are located
        self.patches = {
            'train': load_patches(train_patches),
            'val': load_patches(val_patches)
        }
        self.transforms = {
            'train': train_transform,
            'val': val_transform
        }
        self.train_idx = load_patches(train_idx) if train_idx else None
        self.mode = 'train'
        self.n_channels = n_channels
        self.step = step

    def __getitem__(self, idx):
        [name, top, left, h, w] = self.patches[self.mode][idx]
        top, left, h, w = map(int, [top, left, h, w])
        
        with h5py.File(self.data_path, 'r') as f:
            image = f[name]['data'][top:top + h, left:left + w]
            mask = f[name]['label'][top:top + h, left:left + w]
            image_index = int(name.split('_')[-1].split('.')[0])

            if self.n_channels > 1:
                # Make sure to use the correct directory
                filename_prefix = get_filename_prefix(self.image_dir)
                start_index, end_index = get_image_indices(self.image_dir, filename_prefix)
                full_image = load_2_5D_image(image_index, self.image_dir, self.n_channels, self.step)
                image = full_image[:, top:top + h, left:left + w]
            else:
                image = np.expand_dims(image, axis=0)  # Convert to (1, H, W)

            sample = {'image': image, 'mask': mask}

            if self.transforms[self.mode] is not None:
                sample = self.transforms[self.mode](sample)

            if self.train_idx is not None and self.mode == 'train':
                sample['index'] = self.train_idx[idx]

            return sample

    def __len__(self):
        return len(self.patches[self.mode])

    def train(self):
        """Sets the dataset to training mode."""
        self.mode = 'train'

    def val(self):
        """Sets the dataset to validation mode."""
        self.mode = 'val'

    def set_n_channels(self, n_channels):
        self.n_channels = n_channels

    def set_step(self, step):
        self.step = step




# Update functions to set n_channels, step, and mode
def set_n_channels(dataset, n_channels):
    if isinstance(dataset, HDF52D):
        dataset.set_n_channels(n_channels)
    else:
        raise TypeError("The dataset must be an instance of the HDF52D class.")

def set_step(dataset, step):
    if isinstance(dataset, HDF52D):
        dataset.set_step(step)
    else:
        raise TypeError("The dataset must be an instance of the HDF52D class.")

def set_dataset_mode(dataset, mode):
    if isinstance(dataset, HDF52D):
        dataset.set_mode(mode)
    else:
        raise TypeError("The dataset must be an instance of the HDF52D class.")


if __name__ == '__main__':
    data_path = '/cvdata/yungchen/rdn_revised/data/dataset.hdf5'
    train_patches = '/cvdata/yungchen/rdn_revised/data/patches.csv'
    val_patches = '/cvdata/yungchen/rdn_revised/data/val.csv'
    ratios = '/cvdata/yungchen/rdn_revised/data/ratios.csv'

    transforms = transforms.Compose([dp.Augmentation(output_size=256),
                                     dp.AdjustMask(class_num=3),
                                     dp.Normalize(max=255, min=0)])

    data_set = HDF52D(data_path,train_patches,val_patches,train_transform=transforms, train_idx=ratios, n_channels=n_channels, step=step)
    sample = data_set[1000]
    mask = sample['mask']

    print(np.sum(mask == 0))
    print(np.sum(mask == 1))
    print(np.sum(mask == 2))
    ...
