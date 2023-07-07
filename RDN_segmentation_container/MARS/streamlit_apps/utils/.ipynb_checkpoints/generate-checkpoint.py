import os
import re
import h5py
import random
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import utils.dataprocess as dp
from sklearn.utils import shuffle
from torchvision import transforms
from typing import Union, List
from sklearn.model_selection import train_test_split

#Within this package
from utils.dataset import HDF52D, load_patches

def load_img(path: Union[str, pathlib.Path]):
    path = pathlib.Path(path)
    inputImage = Image.open(str(path)).convert('L')
    return np.array(inputImage)

def sep_string(string: str, target, n=2):
    str_list = string.split(target)
    str_result = str_list[0]
    for idx in range(1, len(str_list) - n):
        str_result = str_result + target + str_list[idx]
    return str_result

def find_match_index(target: str, str_list: List):
    for idx in range(len(str_list)):
        if re.match(target, str_list[idx]):
            return idx
    return None

def generate_hdf5(data_dir: Union[str, pathlib.Path], label_dir: Union[str, pathlib.Path], save_name: str):
    # read the sub file names from a file
    data_dir = pathlib.Path(data_dir)
    label_dir = pathlib.Path(label_dir)
    data_list = os.listdir(str(data_dir))
    label_list = os.listdir(str(label_dir))

    # get the data name
    data_name = [data_list[idx].split(".")[0] for idx in range(len(data_list))]
    label_name = [sep_string(label_list[idx], ".", n=1) for idx in range(len(label_list))]

    # data set_name will give us a list with matched label name and data name
    # for example, if I have a data which name is "XXX.png" and label name is "XXX_3_classes.png"
    # Then dset_name = [{'data': "XXX.png", 'label': "XXX_3_classes.png"}]
    data_set_names = []
    for idx in range(len(label_name)):
        index_name = find_match_index(label_name[idx], data_name)
        if index_name is not None:
            data_set_names.append({'data': data_list[index_name], 'label': label_list[idx]})
        else:
            print(f"The label name {label_name[idx]} may be incorrect.")

    # create hdf5 file
    with h5py.File(save_name, "w") as image_file:
        # load train data to hdf5 file
        for name in data_set_names:
            sample_img = image_file.create_group(sep_string(name['label'], ".", n=1))
            sample_img.create_dataset('data', data=load_img(os.path.join(data_dir, name['data'])))
            sample_img.create_dataset('label', data=load_img(os.path.join(label_dir, name['label'])))

def separate_names(names, train=0.7):
    if train > 1.0:
        train = round(train)
    else:
        train = round(len(names)*train)
    if train > len(names):
        train = len(names)

    #The shuffle method was not randomizing the names
    val_names, train_names = train_test_split(names, test_size=train, shuffle=True)

    print(f"Validating with {len(val_names)} images:\n")
    [print(f"{idx}\n") for idx in val_names]
    return train_names, val_names

def slide_windows(name, shape, output_size=128, stride=32):
    output_size = (output_size, output_size)
    strides = (stride, stride)

    patches_list = []
    idx = 0
    while idx * strides[0] + output_size[0] <= shape[0]:
        top = idx * strides[0]
        j = 0
        while j * strides[1] + output_size[1] <= shape[1]:
            left = j * strides[1]
            patches_list.append([name, top, left,output_size[0], output_size[1]])
            j += 1

        if j * strides[1] < shape[1]:
            left = shape[1] - output_size[1]
            patches_list.append([name, top, left, output_size[0], output_size[1]])
        idx += 1

    if idx * strides[0] < shape[0]:
        top = shape[0] - output_size[0]
        j = 0
        while j * strides[1] + output_size[1] <= shape[1]:
            left = j * strides[1]
            patches_list.append([name, top, left, output_size[0], output_size[1]])
            j += 1

        if j * strides[1] < shape[1]:
            left = shape[1] - output_size[1]
            patches_list.append([name, top, left, output_size[0], output_size[1]])
    return patches_list

def generate_patches(data_path, names, stride=32, output_size=256):

    patches = []
    with h5py.File(data_path, 'r') as data_file:
        for name in names:
            shape = data_file[name]['label'][()].shape
            patches += slide_windows(name, shape, output_size=output_size, stride=stride)
        shuffle(patches)

        for idx in range(len(patches)):
            for j in range(len(patches[idx])):
                patches[idx][j] = str(patches[idx][j])
        return patches

def generate_ratios(data_path, patches_path, class_num=3):
    patches = load_patches(patches_path)
    ratios = []
    with h5py.File(data_path, 'r') as data_file:
        for [name, top, left, h, w] in tqdm(patches, desc="Processing patches:", unit=" Patches"):
            mask = data_file[name]['label'][top: top+h, left: left+w]
            mask = dp.adjustMask(mask, class_num)

            size = 1.0
            for idx in range(len(mask.shape)):
                size *= mask.shape[idx]

            ratio = []
            for idx in range(class_num):
                ratio.append(np.sum(mask == idx)/size)

            ratios.append(ratio)
    return ratios

def get_dirt_bone_patches(patches, ratios):
    ratios = np.array(ratios)
    ratios_idx = np.argsort(-ratios, axis=0)

    # get the patches
    dirt_idx = ratios_idx[:, 1]
    patches = np.asarray(patches)

    ratios_sort = ratios[dirt_idx, :]
    patches_sort = patches[dirt_idx, :]

    dirt_patches = []
    bone_patches = []

    for idx in range(ratios_sort.shape[0]):
        if (ratios_sort[idx, 1] - ratios_sort[idx, 2] > 0.1):
            dirt_patches.append(patches_sort[idx, :].tolist())
        elif (ratios_sort[idx, 2] - ratios_sort[idx, 1] > 0.1):
            bone_patches.append(patches_sort[idx, :].tolist())
        else:
            pass

    bone_index = [1 for i in range(len(bone_patches))]
    dirt_index = [0 for i in range(len(dirt_patches))]

    dirt_patches = shuffle(dirt_patches)
    bone_patches = shuffle(bone_patches)
    dirt_len = len(dirt_patches)
    bone_len = len(bone_patches)
    print(f"There are {bone_len} bone and {dirt_len} dirt patches in the training data...")

    end_idx = dirt_len if dirt_len < bone_len else bone_len

    patches = dirt_patches[0:end_idx] + bone_patches[0:end_idx]
    d_index = dirt_index[0:end_idx] + bone_index[0:end_idx]

    #This is the use of sklearn shuffle, which can't be replaced by the random shuffle
    patches, d_index = shuffle(patches, d_index)
    patches, d_index = shuffle(patches, d_index)
    patches, d_index = shuffle(patches, d_index)
    patches = [[name, int(top), int(left), int(h), int(w)] for [name, top, left, h, w] in patches]
    d_index = [int(idx) for idx in d_index]

    return patches, d_index

def random_patches(dirt_choose_threshold: float, dirt_rate: float, patches: np.array, ratios:np.array):
    # get ratios
    ratios = np.array(ratios)
    ratios_idx = np.argsort(-ratios, axis=0)

    dirt_idx = ratios_idx[:, 1]
    # get the patches
    last_idx = 0
    for i in range(dirt_idx.shape[0]):
        dirt_ratio = ratios[dirt_idx[i], 1]
        if dirt_ratio < dirt_choose_threshold:
            last_idx = i
            break

    dirt_patches_idx = dirt_idx[0:last_idx]
    rest_idx = dirt_idx[last_idx:-1]

    if not (dirt_rate == 0):
        rest_num = round(((last_idx - 1) / dirt_rate) * (1 - dirt_rate))
        if rest_num > rest_idx.shape[0]:
            rest_num = rest_idx.shape[0]
    else:
        rest_num = rest_idx.shape[0]

    random_idx = np.random.choice(rest_idx.shape[0], size=rest_num, replace=False)
    non_dirt_patches_idx = rest_idx[random_idx]

    # choosing patches_idx
    patches_idx = np.concatenate((dirt_patches_idx, non_dirt_patches_idx), axis=0)
    new_patches = np.asarray(patches)[patches_idx, :].tolist()

    new_patches = shuffle(new_patches)
    new_patches = [[name, int(top), int(left), int(h), int(w)] for [name, top, left, h, w] in new_patches]
    return new_patches
