import os
import sys
import pathlib

import cv2

script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(script_dir))
sys.path.append('../../RDN_pytorch/Lib/site_packages/streamlit')
import csv
import glob
import h5py
import math
import uuid
import json
import yaml
import torch
import base64
import random
import shutil
import pickle
import socket
import difflib
import platform
import subprocess
import numpy as np

import pandas as pd
import streamlit as st
import SimpleITK as sitk
import concurrent.futures
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image, ImageColor
from torch import optim
from functools import wraps
from typing import Callable, Tuple
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief
from datetime import datetime, timedelta
from timeit import default_timer as timer
from streamlit.runtime.legacy_caching.hashing import _CodeHasher
from random import shuffle as rand_shuffle
from streamlit.web.server import Server
from sklearn.utils import shuffle as sk_shuffle

from streamlit_apps.streamlit_utils import _get_user
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from streamlit import runtime
#This package
import utils.dataprocess as dp
from net import UNet_Light_RDN
from utils.generate import *
from utils.label_utils import *
from utils.label_utils import _check_label
from streamlit_apps.streamlit_utils import *
from utils.train import rdn_train, rdn_val
from utils.dataset import HDF52D, load_patches, natural_keys
from utils.losses import DomainEnrichLoss, dice_loss, DiceOverlap, Accuracy

import torchvision

from torch.utils.tensorboard import SummaryWriter

#To easily adjust drop down menus
supported_file_types = ["mhd", "nii", "tif", "png", "jpg", "bmp", "dcm"]
slice_types = ["tif", "png", "jpg", "jpeg", "bmp", "dcm"]
volume_types = ["mhd", "mha", "nii", "vtk"]
supported_optimizers = ["Adam", "AdaBelief"]


#Get the small logo for the tab
tab_logo = Image.open(str(script_dir.joinpath('streamlit_apps').joinpath('moon_square.png')))
mars_logo = script_dir.joinpath('streamlit_apps').joinpath('Moon_logo_small.png')

st.set_page_config(page_title="RDN model training",
                   page_icon=tab_logo,
                   layout='wide',
                   initial_sidebar_state='auto')

# #Check the version of streamlit
# version_check = check_streamlit_versions(tested_version=73)
# check_sitk_version()

# if version_check == "below":
#     with st.expander("View/hide warnings"):
#         streamlit_minor(tested_version=73, below_above=version_check)
# if version_check == "above":
#     with st.expander("View/hide warnings"):
#         streamlit_minor(tested_version=73, below_above=version_check)


def main():
    """
    Streamlit GUI to perform data cleaning prior to training. Reading and writing of values is predominantly
    handled with SimpleITK.
    """
    #Get user settings and set up the state
    state = _get_state()
    current_user = _get_user()
    #get_timezone() Maybe but seems to complicated for now

    #### Start of sidebar information
    #Write out the logog to the sidebar with the proper label this uses allow html so it won't work in some situations
    _custom_logo(image_path=mars_logo, image_text="RDN model training")
    # known_user = known_user_dict(user_name=current_user)
    # user_welcome_message(known_user)

    #Get a list of actions to perform in here
    model_radio_list = ['Standardize training data', 'Check training names', 'Data augmentation',
                        'Set up training parameters', 'Finalize data', 'Train model',
                        'Validate model', 'Model gallery']
    query_params = st.experimental_get_query_params()
    default = int(query_params["activity"][0]) if "activity" in query_params else 0

    model_settings_activity = st.sidebar.radio(
        "What are we doing today:",
        model_radio_list,
        index=default
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Common operations:")
    #Open up the path for unsegmented and labels when they become availabel
    if state.unsegmented_training not in [None, "."]:
        if st.sidebar.button("Open unsegmented path", key="99990115"):
            open_windows_explorer(directory=state.unsegmented_training, folder_designation="unsegmented path")

    if state.segmented_training not in [None, "."]:
        if st.sidebar.button("Open label path", key="99990120"):
            open_windows_explorer(directory=state.segmented_training, folder_designation="label path")

    if state.model_validation_directory not in [None, ".", "None"]:
        if st.sidebar.button("Open model path", key="99990128"):
            open_windows_explorer(directory=state.model_validation_directory, folder_designation="model path")
    st.sidebar.markdown("---")

    if st.sidebar.button("Clear settings"):
        if st.checkbox("Are you sure?"):
            state.clear()
    #### End of sidebar information

    #### Start of radio options
    if model_settings_activity == "Standardize training data":
        unsegmented_imgs = []
        segmented_imgs = []
        
        with st.expander("View/hide training data inputs", expanded=True):
            label_col1, label_col2 = st.columns([1, 1])

            with label_col1:
                st.header("New unsegmented training data")
                if str(state.unsegmented_dir) not in [".", None, "None"]:
                    unsegmented_dir = pathlib.Path(st.text_input("Unsegmented directory", state.unsegmented_dir))
                else:
                    unsegmented_dir = pathlib.Path(st.text_input("Unsegmented directory"))

                if str(unsegmented_dir) not in [".", None, "None"]:
                    if unsegmented_dir.exists():
                        state.unsegmented_training = unsegmented_dir.joinpath("new_unseg")
                        st.write(f"Found {unsegmented_dir}")

                        if not unsegmented_imgs:
                            if st.button("Get unsegmented image list"):
                                unsegmented_imgs = glob_flat_list(search_directory=unsegmented_dir,
                                                                  file_types=slice_types,
                                                                  unique=True)
                                state.unsegmented_imgs = unsegmented_imgs

                        if state.unsegmented_imgs:
                            st.write(f"Found {len(state.unsegmented_imgs)} unsegmented images")
                            if st.checkbox("Show unsegemented images"):
                                st.write(state.unsegmented_imgs)
                            if st.button("Clear unsegmented image list"):
                                unsegmented_imgs = []
                                state.unsegmented_imgs = []
                    else:
                        st.error("Couldn't access the directory")
                else:
                    st.write("Please paste in a valid directory")

            with label_col2:
                st.header("New segmented training data")
                if state.segmented_dir and str(state.segmented_dir) != ".":
                    segmented_dir = pathlib.Path(st.text_input("Segmented directory", state.segmented_dir))
                else:
                    segmented_dir = pathlib.Path(st.text_input("Segmented directory"))

                if str(segmented_dir) not in [".", None]:
                    if segmented_dir.exists():
                        state.segmented_dir = segmented_dir
                        state.segmented_training = segmented_dir.joinpath("new_labels")
                        st.write(f"Found {segmented_dir}")

                        if not segmented_imgs:
                            if st.button("Get segmented image list"):
                                segmented_imgs = glob_flat_list(search_directory=segmented_dir,
                                                                file_types=slice_types,
                                                                unique=True)
                                state.segmented_imgs = segmented_imgs

                        if state.segmented_imgs:
                            st.write(f"Found {len(state.segmented_imgs)} unsegmented images")
                            if st.checkbox("Show segemented images"):
                                st.write(state.segmented_imgs)
                            if st.button("Clear segmented image list"):
                                segmented_imgs = []
                                state.segmented_imgs = []
                    else:
                        st.error("Couldn't access the directory")
                else:
                    st.write("Please paste in a valid directory")
        "---"
        if state.unsegmented_imgs and state.segmented_imgs:
            with st.expander("Setup for standardizing labels", expanded=True):
                unseg_dir = pathlib.Path(state.unsegmented_training)
                label_dir = pathlib.Path(state.segmented_training)
                st.write(f"Standardized unsegmented images will be written to {unseg_dir} as tif")
                st.write(f"Standardized labels will be written to {label_dir} as tif")

            if st.button("Standardize training data"):
                unsegmented_imgs = state.unsegmented_imgs
                segmented_imgs = state.segmented_imgs

                #Begin standardization columns
                stnd_col1, stnd_col2 = st.columns([1, 1])

                with stnd_col1:
                    # Convert the unsegmented images to 8bit tif
                    if not unseg_dir.exists():
                        st.write(f"Creating {unseg_dir}...")
                        unseg_dir.mkdir()

                    st.write("Processing unsegmented images...")
                    progress_bar = st.progress(0)
                    img_count = 0
                    for image_file in unsegmented_imgs:
                        clean_image(inputFilename=image_file, suffix="", out_name="", out_type="tif", out_dir=unseg_dir, to_streamlit=True)
                        img_count += 1
                        iteration = np.floor((100 * img_count) / len(unsegmented_imgs))
                        progress_bar.progress(int(iteration))

                with stnd_col2:
                    # Convert the segmented images to 8bit tif
                    if not label_dir.exists():
                        st.write(f"Creating {label_dir}...")
                        label_dir.mkdir()

                    st.write("Processing labels...")
                    progress_bar = st.progress(0)
                    img_count = 0
                    for image_file in segmented_imgs:
                        clean_image(inputFilename=image_file, suffix="", out_name="", out_type="tif", out_dir=label_dir, to_streamlit=True)
                        img_count += 1
                        iteration = np.floor((100 * img_count) / len(segmented_imgs))
                        progress_bar.progress(int(iteration))

                    #Check to make sure the classes are scaled properly
                    check_list = label_dir.rglob("*.tif")
                    st.info("Checking to make sure that there are 3 classes in labels...")

                    progress_bar = st.progress(0)
                    img_count = 0
                    for check in check_list:
                        checking = sitk.ReadImage(str(check))
                        checked = _check_label(inputImage=checking, expected_classes=3)
                        if checked == False:
                            st.write(f"Rescaling {check} to class labels 0, 128, and 255...")
                            rescale_label_proper(input_image=checking, input_file_name=check)
                        img_count += 1
                        iteration = np.floor((100 * img_count) / len(segmented_imgs))
                        #Got to fix this
                        if iteration > 100:
                            iteration = 100
                        progress_bar.progress(int(iteration))
    if model_settings_activity == "Check training names":
        #Makes sure the names match for training and label data
        #state.unsegmented_training is the directory with training data
        #state.segmented_training is the directory with label data

        with st.expander("View/hide training data inputs", expanded=True):
            names_col1, names_col2 = st.columns([1, 1])
            with names_col1:
                st.header("New unsegmented training data")
                if state.unsegmented_training != None :
                    unsegmented_names = glob.glob(str(state.unsegmented_training.joinpath("*.tif")))
                    st.write(f"Found {len(unsegmented_names)} unsegmented training tif files")

            with names_col2:
                st.header("New segmented training data")
                if state.segmented_training != None :
                    segmented_names = glob.glob(str(state.segmented_training.joinpath("*.tif")))
                    st.write(f"Found {len(segmented_names)} segmented training tif files")

        if st.button("Check for inconsistent names") and state.unsegmented_imgs!=None and state.segmented_imgs!=None:
            with st.spinner("Checking for file name consistency between training data and labels..."):
                match_list = segmented_names
                unsegmented_file_list = unsegmented_names
                
                
                st.warning(f"!!! Found {len(match_list)} labels and {len(unsegmented_file_list)} unsegmented images !!!")
                st.info("This may be because the standardization step failed for some reason (unsupported file types),"
                            " or the source folders didn't have the write images. Please check them and try again")
                
                match_list = [pathlib.Path(label_name).parts[-1] for label_name in match_list]
                
                unsegmented_file_list = [pathlib.Path(unseg_name).parts[-1] for unseg_name in unsegmented_file_list]
                state.unsegmented_file_list = unsegmented_file_list
                state.match_list = match_list
               
                #We only want to consider file names that don't have a match
                match_list = [label_name for label_name in match_list if label_name not in unsegmented_file_list]
                with names_col1:
                    st.write("Unsegmented names:")
                    st.write(state.unsegmented_file_list)

                with names_col2:
                    st.write("Label names:")
                    st.write(state.match_list)
               
                if match_list:
                    state.match_list = match_list
                    with st.expander("View/hide match information", expanded=False):
                        unmatched_labels = {}
                        
                        for label_name in match_list:
                            if not state.unsegmented_training.joinpath(label_name).is_file():
                                file_type_check = glob.glob(str(state.unsegmented_training.joinpath(label_name.rsplit(".")[0])))
                                if len(file_type_check) == 1:
                                    unmatched_labels[str(label_name)] = file_type_check
                                elif len(file_type_check) > 1:
                                    st.warning(f"Multiple matches for {label_name}")
                                    st.write(file_type_check)
                                    unmatched_labels[str(label_name)] = file_type_check
                                else:
                                    possible_match = check_for_match(missing_file=label_name,
                                                                     check_filelist=unsegmented_file_list)
                                    if possible_match == None:
                                        unmatched_labels[str(label_name)] = ["None"]
                                    else:
                                        unmatched_labels[str(label_name)] = possible_match

                        if len(unmatched_labels) == 0:
                            st.write("No misnamed files found! :balloons:")
                            state.unmatched_labels = pd.DataFrame()
                        else:
                            unmatched_labels = pd.DataFrame.from_dict(unmatched_labels,  orient='index')
                            unmatched_labels.reset_index(drop=False, inplace=True)
                            if unmatched_labels.shape[1] > 2:
                                columns_names = [f"unsegmented_name_match_{column_num}" for column_num in unmatched_labels.columns[1:]]
                                columns_names = ["label_name"] + columns_names
                                unmatched_labels.columns = columns_names
                            else:
                                unmatched_labels.columns = ["label_name", "unsegmented_name_match"]
                    if len(unmatched_labels) != 0:
                        state.unmatched_labels = unmatched_labels
                else:
                    st.header(":smile: :rainbow: No misnamed files found! :rainbow: :smile:")
                    st.balloons()
        elif(state.unsegmented_imgs==None and state.segmented_imgs!=None):
            st.warning(f"You need to put good unsegmented training data")
        elif(state.unsegmented_imgs!=None and state.segmented_imgs==None):
            st.warning(f"You need to put good segmented training data")
        elif(state.unsegmented_imgs==None and state.segmented_imgs==None):
            st.warning(f"You need to put good unsegmented and segmented training data")
        #This may be a bad idea but it works for now
        try:
            if state.unmatched_labels == None:
                st.empty()
        except ValueError:
            if not state.unmatched_labels.empty:
                if len(state.unmatched_labels) != 0:
                    st.warning("There are labels with unsegmented tif files that don't match!")
                    if st.checkbox("View labels with matching unsegmented names!"):
                        df = state.unmatched_labels
                        st.write(df)
                        if "None" in df.unsegmented_name_match.values:
                            st.info("If 'None' is present in the second column it means there was no easy match found "
                                    "for that label. You will have to manually rename the label or make sure it was "
                                    "standardized")
                            if st.button("Open label path", key="99990301"):
                                #Probably should be a function
                                command_string = f"explorer {state.segmented_dir.joinpath('new_labels')}"
                                with st.spinner(f"Opening label folder: {state.segmented_dir.joinpath('new_labels')}"):
                                    subprocess.Popen(command_string, shell=True)
                                    remaining_choices_list = [item for item in state.unsegmented_file_list if item not in state.match_list]
                                    st.write("Remaining unsegmented training data file names:", remaining_choices_list)

                        elif df.shape[1] > 2:
                            st.info("Multiple possible matches were found, so you will have to manually rename some before proceding.")
                            if st.button("Open label path", key="99990305"):
                                command_string = f"explorer {state.segmented_dir.joinpath('new_labels')}"
                                with st.spinner(f"Opening label folder: {state.segmented_dir.joinpath('new_labels')}"):
                                    subprocess.Popen(command_string, shell=True)
                                    remaining_choices_list = [item for item in state.unsegmented_file_list if item not in state.match_list]
                                    st.write("Remaining unsegmented training data file names:", remaining_choices_list)
                        else:
                            if st.button("Rename labels to match the unsegmented tif!"):
                                for row in df.itertuples():
                                    if row.unsegmented_name_match != "None":
                                        label_path = state.segmented_dir.joinpath("new_labels")
                                        rename_from = label_path.joinpath(row.label_name)
                                        rename_to = label_path.joinpath(row.unsegmented_name_match)
                                        st.write(f"Renaming {row.label_name}")
                                        if rename_from.is_file():
                                            try:
                                                rename_from.rename(rename_to)
                                            except FileExistsError:
                                                st.error(f"Whoops! It looks like {rename_to} already exists.")


                                        else:
                                            st.error(f"Whoops! Looks like {rename_from} either moved or we can't access it.")
                                            st.write([label_path.as_posix()])
                                state.unmatched_labels = None
                                st.info("Renaming done, please rerun the label check to see if there are remaining issues!")
    
    if model_settings_activity == "Data augmentation":
        #state.unsegmented_training is the directory with training data
        #state.segmented_training is the directory with label data

        with st.expander("View/hide infomration", expanded=False):
            st.info("This is an optional step where we explicitly create variation in our training data."
                    "Althoughr there are internal steps that randomize data augemntation (e.g. skewing, invert, cropping), "
                    "this is still useful if some of the images you are working with have very low grey values.")

        data_aug_1, data_aug_2 = st.columns([1, 1])
        with data_aug_1:
            if state.unsegmented_imgs!=None and state.segmented_imgs!=None:
                st.header("New unsegmented training data")
                training_list = [item.name for item in state.unsegmented_training.rglob("*.tif")]
                training_list = [item for item in training_list if "_rescaled.tif" not in item]
                training_list = [item for item in training_list if "_downscaled.tif" not in item]
                st.write(training_list)

        with data_aug_2:
            if state.unsegmented_imgs!=None and state.segmented_imgs!=None:
                st.header("New segmented training data")
                label_list = [item.name for item in state.segmented_training.rglob("*.tif")]
                label_list = [item for item in label_list if "_rescaled.tif" not in item]
                label_list = [item for item in label_list if "_downscaled.tif" not in item]
                st.write(label_list)

        if st.button("Rescale training data") and state.unsegmented_imgs!=None and state.segmented_imgs!=None:
            progress_bar = st.progress(0)
            img_count = 0
            for rescale in training_list:
                up_check = rescale.replace(".tif", "_rescaled.tif")
                down_check = rescale.replace(".tif", "_downscaled.tif")
                out_dir = state.unsegmented_training
                label_dir = state.segmented_training

                if not state.unsegmented_training.joinpath(up_check).exists():
                    st.write(f"Upscaling {rescale}")
                    rescale_intensity(inputFilename=out_dir.joinpath(rescale),
                                      writeOut=True,
                                      file_type="tif",
                                      outDir=out_dir)
                    shutil.copy(str(label_dir.joinpath(rescale)), str(label_dir.joinpath(up_check)))

                if not state.unsegmented_training.joinpath(down_check).exists():
                    st.write(f"Downscaling {rescale}")
                    downscale_intensity(inputFilename=out_dir.joinpath(rescale),
                                        downscale_value=255,
                                        writeOut=True,
                                        file_type="tif",
                                        outDir=out_dir)
                    shutil.copy(str(label_dir.joinpath(rescale)), str(label_dir.joinpath(down_check)))
                img_count += 1
                iteration = np.floor((100 * img_count) / len(training_list))
                progress_bar.progress(int(iteration))
        elif(state.unsegmented_imgs==None and state.segmented_imgs!=None):
            st.warning(f"You need to put good unsegmented training data")
        elif(state.unsegmented_imgs!=None and state.segmented_imgs==None):
            st.warning(f"You need to put good segmented training data")
        elif(state.unsegmented_imgs==None and state.segmented_imgs==None):
            st.warning(f"You need to put good unsegmented and segmented training data")



    if model_settings_activity == "Set up training parameters":
        # Set up the inputs for this section
        # state.unsegmented_training is the directory with training data
        # state.segmented_training is the directory with label data

        #Where we will write the hdf5, yaml config, and model iterations
        if state.unsegmented_imgs!=None and state.segmented_imgs!=None:
            data_path = state.unsegmented_training.parent.parent.joinpath("data")
            train_yaml = str(script_dir.joinpath("yaml").joinpath("train.yaml"))
            test_yaml = str(script_dir.joinpath("yaml").joinpath("test.yaml"))
            new_models_path = data_path.joinpath("new_model")

            #Read in the default yaml information
            train_yaml_file = read_train_yaml(yaml_file=train_yaml)
            test_yaml_file = read_train_yaml(yaml_file=test_yaml)

            #Get the GPU device
            device_num = initiate_cuda()

            #Check if we can set the GPU and return various error messagges if we can't.
            state.device_num = setup_gpu(device_num=device_num, state=state)

            #Get the information for the user so they can see
            cuda_mem = int(torch.cuda.get_device_properties(device=state.use_gpu).total_memory)
            cuda_mem = list(_convert_size(sizeBytes=cuda_mem))

            with st.expander("View/hide information"):
                st.info(
                    f"GPU set to device number {state.use_gpu}: {torch.cuda.get_device_properties(device=state.use_gpu).name}, "
                    f"{cuda_mem[0]} of VRAM")
                if float(cuda_mem[1]) < 5.0:
                    st.error("The amount of VRAM that you have makes training impossible. Sorry. :sob:")
                elif float(cuda_mem[1]) < 8.0:
                    st.warning("The amount of VRAM that you have is going to make this very challenging. :frowning:")
                st.info(f"Training data save path: {data_path}")
                st.info(f"New models will be saved to: {new_models_path}")

            #Grab the information from the default yamls.
            previous_batch = train_yaml_file["data_loader"]["batch_size"]
            previous_period = train_yaml_file["period"]
            previous_epoch = train_yaml_file["train_param"]["Epoch"]
            previous_learning_rate = train_yaml_file["optimizer"]["lr"]
            previous_weight_decay = train_yaml_file["optimizer"]["weight_decay"]

            #Columns for the standard settings
            yaml_col_1, yaml_col_2, yaml_col_3, yaml_col_4 = st.columns([1, 1, 1, 2])
            with yaml_col_1:
                batch_size = st.text_input("Input batch size (an integer):", f"{previous_batch}")
            with yaml_col_3:
                epochs_num = st.text_input("Input epochs (an integer):", f"{previous_epoch}")
            with yaml_col_4:
                if st.checkbox("Train from a previously trained model?"):
                    state.train_from_previous = True
                    previous_model = st.file_uploader("Select model", type=["pth"], accept_multiple_files=False)
                    try:
                        pre_trained = torch.load(previous_model, map_location=f'cuda:{int(state.use_gpu)}')
                    except AttributeError:
                        st.info("Please navigate to or drop a .pth file above")
                else:
                    state.train_from_previous = False

            #If there are additional optimizer settings then they go under advanced use
            if st.checkbox("Advanced parameters"):
                yaml2_col_1, yaml2_col_2, yaml2_col_3, yaml2_col_4 = st.columns([1, 1, 1, 1])
                with yaml2_col_1:
                    optimizer = st.selectbox("Optimizer", supported_optimizers)
                with yaml2_col_2:
                    #Adam is the default, but any can be added to the list above and then settings can be passed over.
                    if optimizer == "AdaBelief":
                        learning_rate = st.text_input("Optimizer learning rate", f"{1e-03:.8f}")
                        learning_rate = f"{float(learning_rate):.1e}"
                    else:
                        learning_rate = st.text_input("Optimizer learning rate", previous_learning_rate)
                with yaml2_col_3:
                    if optimizer == "AdaBelief":
                        epsilon = st.text_input("Optimizer epsilon ", f"{1e-16:.16f}")
                        epsilon = f"{float(epsilon):.1e}"
                    else:
                        weight_decay = st.text_input("Optimizer weight decay", f"{previous_weight_decay:.8f}")
                        weight_decay = f"{float(weight_decay):.1e}"
                with yaml2_col_4:
                    if optimizer == "AdaBelief":
                        st.write("Weight decay is decoupled. Recommended epsilon:")
                        st.write(f"Between 1e-08 and 1e-16")
                        st.info(f"Current: {epsilon}")

                with yaml_col_2:
                    if optimizer == "AdaBelief":
                        period_size = False
                        "Step size is algorithmically controlled."
                    else:
                        period_size = st.text_input("Input period (an integer):", f"{previous_period}")
            else:
                optimizer = "Adam"
                learning_rate = previous_learning_rate
                weight_decay = previous_weight_decay
            if st.button("Commit changes"):
                state.data_path = data_path
                if not data_path.exists():
                    data_path.mkdir()

                if state.train_from_previous:
                    pretrained_path = data_path.joinpath("pretrained_model")
                    if not pretrained_path.exists():
                        pretrained_path.mkdir()
                    model_name = str(pretrained_path.joinpath("pretrained_model.pth").as_posix())
                    torch.save(pre_trained, str(model_name))
                    train_yaml_file["model"]["if_pre_train"] = "true"
                    train_yaml_file["model"]["path"] = str(model_name)
                else:
                    train_yaml_file["model"]["if_pre_train"] = "false"
                    train_yaml_file["model"]["path"] = None

                #GPU device index
                train_yaml_file["gpu_config"]["gpu_name"] = int(state.use_gpu)

                #Loops
                train_yaml_file["data_loader"]["batch_size"] = int(batch_size)
                if period_size:
                    train_yaml_file["period"] = int(period_size)
                else:
                    train_yaml_file["period"] = None
                train_yaml_file["train_param"]["Epoch"] = int(epochs_num)

                #Optimizer paramters
                train_yaml_file["optimizer"]["method"] = f'{optimizer}'
                train_yaml_file["optimizer"]["lr"] = float(learning_rate)
                if optimizer == "AdaBelief":
                    train_yaml_file["optimizer"]["epsilon"] = epsilon
                    train_yaml_file["optimizer"]["weight_decay"] = "false"
                else:
                    train_yaml_file["optimizer"]["weight_decay"] = float(weight_decay)

                #Data path
                train_yaml_file["path"]["data_path"] = str(data_path.joinpath("dataset.hdf5").as_posix())
                train_yaml_file["path"]["save_path"] = str(new_models_path.as_posix())

                #CSV path
                train_yaml_file["csv_path"]["train"] = str(data_path.joinpath("patches.csv").as_posix())
                train_yaml_file["csv_path"]["val"] = str(data_path.joinpath("val.csv").as_posix())
                train_yaml_file["csv_path"]["ratios"] = str(data_path.joinpath("ratios.csv").as_posix())

                #Test yaml
                test_yaml_file["gpu_config"]["gpu_name"] = int(state.use_gpu)
                test_yaml_file["path"]["data_path"] = str(data_path.joinpath("dataset.hdf5").as_posix())
                test_yaml_file["model"]["path"] = str(new_models_path.as_posix())

                test_yaml_file["csv_path"]["val"] = str(data_path.joinpath("val.csv").as_posix())

                new_yaml_path = data_path.joinpath("yaml")
                new_train_yaml_name = new_yaml_path.joinpath("train.yaml")
                new_test_yaml_name = new_yaml_path.joinpath("test.yaml")
                if not new_yaml_path.exists():
                    new_yaml_path.mkdir()

                with open(str(new_train_yaml_name), 'w') as f:
                    yaml.dump(train_yaml_file, f)
                    state.new_train_yaml_name = new_train_yaml_name

                with open(str(new_test_yaml_name), 'w') as f:
                    yaml.dump(test_yaml_file, f)
                    state.new_test_yaml_name = new_test_yaml_name
                st.info(f"Training and validation parameters written to {new_yaml_path}")
        elif(state.unsegmented_imgs==None and state.segmented_imgs!=None):
            st.warning(f"You need to put good unsegmented training data")
        elif(state.unsegmented_imgs!=None and state.segmented_imgs==None):
            st.warning(f"You need to put good segmented training data")
        elif(state.unsegmented_imgs==None and state.segmented_imgs==None):
            st.warning(f"You need to put good unsegmented and segmented training data")

    if model_settings_activity == "Finalize data":
        state = _get_state()
        if state.data_path in [None, "None", "."]:
            st.warning("Dataset path not defined. If you're looking to pick up where you left off, go back to "
                       "Setting up training parameters.")
        elif state.unsegmented_imgs!=None and state.segmented_imgs!=None:
            training_data_dir = state.unsegmented_training.as_posix()
            training_label_dir = state.segmented_training.as_posix()
            hdf5_name = str(state.data_path.joinpath("dataset.hdf5").as_posix())
            patches_name = str(state.data_path.joinpath("patches.csv").as_posix())
            val_name = str(state.data_path.joinpath("val.csv").as_posix())
            ratios_name = str(state.data_path.joinpath("ratios.csv").as_posix())

            class_num = 3 #Maybe more or less classes later.

            st.info(f"Training data directory: {training_data_dir}")
            st.info(f"Training label directory: {training_label_dir}")
            st.info(f"Training label directory: {hdf5_name}")
            ("---")

            if st.button("Generate HDF5 dataset"):
                generate_hdf5_streamlit(data_dir=state.unsegmented_training,
                            label_dir=state.segmented_training,
                            save_name=hdf5_name)
                state.hdf5 = hdf5_name
            ("---")

            train_col_1, train_col_2, train_col_3, train_col_4 = st.columns([1, 1, 2, 1])
            with train_col_1:
                stride = int(st.text_input("Stride size:", 32))
            with train_col_2:
                train_size = float(st.text_input("Ratio of train/test split:", 0.7))
            with train_col_3:
                if st.checkbox("Exclude images from validation set?"):
                    state.exclude_from_split = True
                    st.always_train = st.text_input("CSV file with names to exlude", )
                else:
                    state.exclude_from_split = False
                    st.always_train = None
            
            if st.button("Generate patches and validation data"):
                output = 64
               
                val_names = generate_patches_streamlit(hdf5_file=hdf5_name, patches_csv=patches_name,
                                                    validation_csv=val_name, train_ratio=train_size,
                                                    stride=stride, output_size=output, always_train_csv=False)
                state.val_names = val_names
            if state.val_names not in [None, "None"]:
                if st.checkbox("View validation image set:"):
                    st.write(state.val_names)
            ("---")

            ratio_col_1, ratio_col_2, ratio_col_3, ratio_col_4 = st.columns([2, 2, 1, 1])

            with ratio_col_2:
                if st.checkbox("Multithreaded calculation (recommended)"):
                    state.ratios_parallel = True
                else:
                    state.ratios_parallel = False

                if state.ratios_parallel:
                    cpus_avail = list(range(cpu_count() + 1))
                    suggested_cpu = cpu_count() - 1
                    state.num_threads = st.selectbox("Select number of cores to use",
                                                    cpus_avail, index=suggested_cpu)

            with ratio_col_1:
                if st.button("Calculate class ratios from dataset"):
                    start_multi = timer()
                    if state.ratios_parallel:
                        patches_df = _setup_patches(patches_csv=patches_name)
                        new_ratios = parallelize_ratios(df=patches_df, func=generate_ratios_streamlit_multi,
                                                        hdf5_file=hdf5_name, class_num=3, n_cores=state.num_threads)
                    else:
                        ratios = generate_ratios_streamlit(hdf5_file=hdf5_name, patches_csv=patches_name,
                                                        class_num=class_num)
                        new_ratios = pd.DataFrame(ratios)
                        ratio_headers = [f"Class {idx}" for idx in range(class_num)] #Just in case we do increase class numbers
                        new_ratios.columns = ratio_headers

                    _end_timer(start_timer=start_multi, message="Calculation")
                    class_means = pd.DataFrame(new_ratios.mean()).T
                    class_means.columns = ["Air", "Non-Bone", "Bone"]
                    st.write("Training data class percentages:")
                    st.write(class_means)
                    new_ratios.to_csv(f"{ratios_name}", index=False)
        elif(state.unsegmented_imgs==None and state.segmented_imgs!=None):
            st.warning(f"You need to put good unsegmented training data")
        elif(state.unsegmented_imgs!=None and state.segmented_imgs==None):
            st.warning(f"You need to put good segmented training data")
        elif(state.unsegmented_imgs==None and state.segmented_imgs==None):
            st.warning(f"You need to put good unsegmented and segmented training data")


    if model_settings_activity == "Train model":
        if state.data_path in [None, "None", "."]:
            st.warning("Dataset path not defined. If you're looking to pick up where you left off, go back to "
                       "Setting up training parameters.")
        else:
            new_yaml_path = state.data_path.joinpath("yaml")
            new_train_yaml_name = new_yaml_path.joinpath("train.yaml")
            new_test_yaml_name = new_yaml_path.joinpath("test.yaml")

            #Should probably move this up to the yaml section and put it into state
            timestamp = time.time()
            sub_save_file = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H')
            config_col_1, config_col_2, config_col_3, config_col_4 = st.columns([1, 1, 1, 2])

            if st.checkbox("Load training configuration!"):
                with open(new_train_yaml_name) as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)

                #GPU device index
                gpu_ID = config["gpu_config"]["gpu_name"]

                #Previous model
                if config["model"]["if_pre_train"] == "true":
                    model_start = config["model"]["path"]
                    state.train_from_previous = True
                else:
                    model_start = False
                    state.train_from_previous = False

                #Loops
                batch_size = config["data_loader"]["batch_size"]
                periods = config["period"]
                num_epochs = config["train_param"]["Epoch"]

                #Optimizer paramters
                optimize_method = config["optimizer"]["method"]
                optimize_learning_rate = config["optimizer"]["lr"]
                optimize_learning_weight = config["optimizer"]["weight_decay"]

                #Data path
                train_data = config["path"]["data_path"]
                model_output = config["path"]["save_path"]

                #CSV path
                patch_csv = config["csv_path"]["train"]
                validation_csv = config["csv_path"]["val"]
                ratio_csv = config["csv_path"]["ratios"]

                with config_col_1:
                    st.header("Graphics card")
                    st.info(f"GPU Training ID: {gpu_ID}")

                with config_col_2:
                    st.header("Batch info.")
                    st.info(f"Batch size: {batch_size}")
                    st.info(f"Periods:    {periods}")
                    st.info(f"Epochs:     {num_epochs}")

                with config_col_3:
                    st.header("Optimization")
                    st.info(f"Method: {optimize_method}")
                    st.info(f"Learning rate: {optimize_learning_rate}")
                    if optimize_method == "AdaBelief":
                        st.info(f"Weight decay: Decoupled")
                        epsilon = config["optimizer"]["epsilon"]
                        st.info(f"Epsilon: {epsilon}")
                    else:
                        st.info(f"Weight decay: {optimize_learning_weight}")

                with config_col_4:
                    st.header("Training files")
                    if pathlib.Path(patch_csv).exists():
                        st.info(f"Patches: {patch_csv}")
                    else:
                        st.error("Can't find patch csv! Did you generate it?")

                    if pathlib.Path(ratio_csv).exists():
                        st.info(f"Class ratios: {ratio_csv}")
                    else:
                        st.error("Can't find ratio csv! Did you generate it?")

                    if pathlib.Path(validation_csv).exists():
                        st.info(f"Validation: {validation_csv}")
                    else:
                        st.error("Can't find validation csv! Did you generate it?")


                model_save_path = pathlib.Path(config['path']['save_path'])
                save_path = model_save_path.joinpath(sub_save_file)
                state.save_path = save_path

                if model_start != False:
                    st.info(f"Starting training from model {model_start}")
                st.info(f"Models will be written to {save_path.as_posix()}")


                #If the folders don't exist we make them
                if not model_save_path.exists():
                    st. info(f"Making {model_save_path}")
                    model_save_path.mkdir()

                if not save_path.exists():
                    st.info(f"Making {save_path}")
                    save_path.mkdir()

            # get model
            if st.checkbox("Intialize UNet"):
                # check if gpu is available
                if config['gpu_config']['use_gpu']:
                    torch.cuda.set_device(config['gpu_config']['gpu_name'])  # '1','0'

                net = UNet_Light_RDN(n_channels=config['model']['n_channels'], n_classes=config['model']['class_num'])
                if state.train_from_previous:
                    if config['model']['path'] is not None:
                        if config['gpu_config']['use_gpu']:
                            net.load_state_dict(torch.load(config['model']['path'],
                                                        map_location=torch.device(type='cuda',
                                                                                    index=config['gpu_config']['gpu_name'])))
                        else:
                            net.load_state_dict(torch.load(config['model']['path']))
                state.net = net
                if str(config['optimizer']['method']) == "AdaBelief":
                    optimizer = AdaBelief(net.parameters(),
                                        lr=float(config['optimizer']['lr']),
                                        eps=float(config["optimizer"]["epsilon"]),
                                        betas=(0.9, 0.999),
                                        weight_decouple=True,
                                        rectify=False)
                else:
                    optimizer = getattr(optim,
                                        config['optimizer']['method'])(net.parameters(),
                                                                    lr=config['optimizer']['lr'],
                                                                    weight_decay=config['optimizer']['weight_decay'])
                    #learning rate schedule
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['period'], gamma=0.1)


            
                # save the yaml file to savepath
                current_config = str(pathlib.Path(state.save_path).joinpath('Config.yaml'))
                st.write(f"Saving session configuration to {current_config}")
                with open(current_config, 'w') as file:
                    yaml.dump(config, file)

                # get training Epoch
                Epoch = config['train_param']['Epoch']

                # load patches and ratios
                train_patches = load_patches(config['csv_path']['train'])
                val_patches = load_patches(config['csv_path']['val'])
                ratios = load_patches(config['csv_path']['ratios'])
                period = config['period']
                if period == None:
                    # This gets used later to ramp up the amount of non-bone that is being thrown into the training.
                    # May have to think of a fancy way to get an equivelant number with Adabelief, but it is being set
                    # to the default Adam period for now.
                    period = 8
                
                # create train transform
                train_transform = transforms.Compose([dp.Augmentation(output_size=64), #config['output_size']
                                                    dp.AdjustMask(class_num=config['model']['class_num']),
                                                    dp.Normalize(max=255, min=0),
                                                    dp.ToTensor()])
                val_transform = transforms.Compose([dp.AdjustMask(class_num=config['model']['class_num']),
                                                    dp.Normalize(max=255, min=0),
                                                    dp.ToTensor()])
                # if st.button("Launch Tensorboard !"):
                    
                #     subprocess.call('tensorboard --logdir=runs', shell=True)
                #     st.write(f"TensorBoard 2.10.0 at http://localhost:6006/#timeseries")
                    
                if st.button("Train model!"):
                    # training

                    progress_bar = st.progress(0)
                    epoch_count = 0
                    #st.write(f"TensorBoard is availaible, run this following command in a terminal : tensorboard --logdir=runs")
                    st.sidebar.write(f"Epoch progress:")
                    st.write(f"Progress training {Epoch} epochs...")
                    total_timer = timer()
                    iteration = 0
                    nb_ite = 0
                    #subprocess.call('echo "TensorBoard available, run this command to enable it : tensorboard --logdir=runs"', shell=True)
                    for i_epoch in range(Epoch):
                        st.sidebar.write(f"Epoch {epoch_count + 1} of {Epoch}")
                        if i_epoch < period:
                            #dirt_rate = 0.5
                            air_rate = 0.1
                        elif i_epoch < 2 * period and i_epoch >= period:
                            #dirt_rate = 0.3
                            air_rate = 0.2
                        elif i_epoch < 3 * period and i_epoch >= 2 * period:
                            #dirt_rate = 0.1
                            air_rate = 0.4
                        else:
                            #dirt_rate = 0.0
                            air_rate = 0.5

                        #Get patches 
                        patches = get_minimum_dirt_patches(dirt_choose_threshold=0.1, dirt_rate=0,
                                                   patches=train_patches, ratios=ratios)

                        DEB_patches, index = get_dirt_bone_patches(train_patches, ratios, air_rate)

                        data_set = HDF52D(config['path']['data_path'], patches, val_patches,
                                        train_transform=train_transform,
                                        val_transform=val_transform)

                        DEB_data_set = HDF52D(config['path']['data_path'], DEB_patches, val_patches,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        train_idx=index)

                        train_data_loader = []

                        current_batch = int(config['data_loader']['batch_size'])

                        
                        # train_data_loader.append(DataLoader(dataset=training_data_set,
                        #                                     batch_size=current_batch,
                        #                                     shuffle=True,
                        #                                     num_workers=0))


                        train_data_loader.append(DataLoader(dataset=DEB_data_set,
                                                            batch_size=current_batch,
                                                            shuffle=True,
                                                            num_workers=0))
                        
                        train_data_loader.append(DataLoader(dataset=DEB_data_set,
                                                            batch_size=current_batch,
                                                            shuffle=True,
                                                            num_workers=0))
                        
                        print(f"learning rate {optimizer.param_groups[0]['lr']:.6f}")

                        nb_ite = rdn_train(net, optimizer, train_data_loader, epoch=i_epoch,
                                total_epoch=Epoch, use_gpu=config['gpu_config']['use_gpu'], tensorboard_plot=True, nb_ite=nb_ite)
                        #lr_scheduler.step()

                        # validating
                        val_loss, class_val = rdn_val(net, data_set,
                                                    use_gpu=config['gpu_config']['use_gpu'],
                                                    i_epoch=i_epoch,
                                                    class_num=config['model']['class_num'])

                        # save model
                        save_name = state.save_path.joinpath(f"Loss-{epoch_count}_{val_loss:.6f}.pth")
                        torch.save(net.state_dict(), save_name)
                        class_val = pd.DataFrame(class_val)
                        class_val.columns = ["Class Dice overlap"]
                        st.sidebar.write(class_val)
                        epoch_count += 1
                        iteration = np.floor((100 * epoch_count) / int(Epoch))
                        progress_bar.progress(int(iteration))
                    st.balloons()
                    st.info(':joy: :rainbow: Training is finished! :rainbow: :joy:')
                    _end_timer(start_timer=total_timer, message="Total training of model")

    if model_settings_activity == "Validate model":
        if state.data_path in [None, "None", "."]:
            st.warning("Dataset path not defined. If you're looking to pick up where you left off, go back to "
                       "Setting up training parameters.")
        else:
            new_yaml_path = state.data_path.joinpath("yaml")
            new_test_yaml_name = new_yaml_path.joinpath("test.yaml")

            criterion_df = np.array([])
            # class_overlap = pd.array([])
            class_overlap = pd.DataFrame()
            if st.checkbox("Load validation parameters"):
                validate_yaml = read_test_yaml(str(new_test_yaml_name))
                state.validate_yaml = validate_yaml

                #GPU device index
                gpu_ID = validate_yaml["gpu_config"]["gpu_name"]

                #Data path for the hdf5
                train_data = validate_yaml["path"]["data_path"]
                validation_csv = validate_yaml["csv_path"]["val"]

                config_col_1, config_col_2, config_col_3 = st.columns([1, 1, 2])
                with config_col_1:
                    st.header("Graphics card")
                    st.info(f"GPU Validation ID: {gpu_ID}")

                with config_col_2:
                    st.empty()

                with config_col_3:
                    st.header("Validation files")
                    if pathlib.Path(validation_csv).exists():
                        st.info(f"CSV: {validation_csv}")
                    else:
                        st.error("Can't find validation csv! Did you generate it?")
                    if pathlib.Path(train_data).exists():
                        st.info(f"Dataset: {train_data}")
                    else:
                        st.error("Can't find validation data, is the path correct?")

                model_directory = str(train_data).replace("dataset.hdf5", "new_model")
                with st.expander("View hide/validation choices", expanded=True):
                    model_directory = [str(e) for e in pathlib.Path(model_directory).iterdir() if e.is_dir()]
                    model_directory.sort(key=os.path.getctime, reverse=True)
                    model_validation_directory = st.selectbox("Select model directory (Likely the most recent)",
                                                            options=model_directory,
                                                            index=0)
                    if pathlib.Path(model_validation_directory).is_dir():
                        state.model_validation_directory = model_validation_directory
                        model_list = glob.glob(str(pathlib.Path(model_validation_directory).joinpath("*.pth")))
                        model_list.sort(key=natural_keys)
                        st.info(f"Found {len(model_list)} models to validate")

            # parsing the input parameter
            if st.button("Validate"):
                save_path = pathlib.Path(str(model_validation_directory))
                criterion_output = save_path.joinpath("Model_scores.csv")
                class_output = save_path.joinpath("Class_overlap.csv")

                # get the config file
                config = validate_yaml

                # check if gpu is available
                if config['gpu_config']['use_gpu']:
                    torch.cuda.set_device(config['gpu_config']['gpu_name'])  # '1','0'

                # get nets' name list and sort by creation time
                # create train transform
                val_transform = transforms.Compose([dp.AdjustMask(class_num=config['model']['class_num']),
                                                    dp.Normalize(max=255, min=0),
                                                    dp.ToTensor()])
                data_set = HDF52D(config['path']['data_path'], [], config['csv_path']['val'], val_transform=val_transform)
                data_set.val()

                progress_bar = st.progress(0)
                model_count = 0
                num_models = len(model_list)
                state.model_list = model_list
                for val_model in model_list:
                    model_path = model_validation_directory

                    # get model
                    net = UNet_Light_RDN(n_channels=config['model']['n_channels'], n_classes=config['model']['class_num'])
                    if config['gpu_config']['use_gpu']:
                        net.load_state_dict(torch.load(val_model,
                                                    map_location=torch.device(type='cuda',
                                                                                index=config['gpu_config']['gpu_name'])))
                    else:
                        net.load_state_dict(torch.load(val_model))

                    print(f'Model Path: {model_path}.')
                    criterion, class_o = rdn_val(net, data_set,
                                                use_gpu=config['gpu_config']['use_gpu'],
                                                class_num=config['model']['class_num'])
                    print(f"Total score: {criterion}\n")
                    class_o = pd.DataFrame([class_o], index=[f"Model_{model_count}"]).T
                    criterion_df = np.hstack((criterion_df, criterion))
                    class_overlap = pd.concat([class_overlap, class_o], axis=1)
                    model_count += 1
                    iteration = np.floor((100 * model_count) / int(num_models))
                    progress_bar.progress(int(iteration))

                # Save the Dice Overlap scores.
                criterion_df = pd.DataFrame(criterion_df)
                criterion_df.columns = ["Combined_Dice_overlap"]
                criterion_df.sort_values(by=["Combined_Dice_overlap"], ascending=False, inplace=True)
                criterion_df.to_csv(str(criterion_output))

                class_overlap = pd.DataFrame(class_overlap).T
                class_overlap.columns = ["Air", "Non_Bone", "Bone"]
                class_overlap.sort_values(by=["Non_Bone", "Bone"], ascending=False, inplace=True)
                save_path.joinpath("Classes.csv")
                class_overlap.to_csv(str(class_output))
                state.class_overlap_df = class_overlap

                st.write('Validating is finished.')

                st.balloons()
                validation_col1, validation_col2 = st.columns([1, 1])
                with validation_col1:
                    st.subheader(f"Best 3 models per non-bone class:")
                    st.write(class_overlap.head(3))
                with validation_col2:
                    st.subheader(f"Overall Dice overlap:")
                    st.write(criterion_df.head(3))
                state.validated_models = True

            #Save the model so people can name it whatever they want
            if state.validated_models:
                model_col_1, model_col_2, model_col_3 = st.columns([1, 1, 1])
                #Give the top 10 choices, because why not?
                model_list = list(state.class_overlap_df.index[:10])
                with model_col_1:
                    save_model = st.selectbox("Select the model to save?", model_list)
                    model_index = save_model.replace("Model_", "Loss-")
                    search_directory = pathlib.Path(state.model_validation_directory)
                    selected_model = glob.glob(str(search_directory.joinpath(f"{model_index}_*.pth")))[0]

                with model_col_2:
                    if pathlib.Path(selected_model).is_file():
                        st.write("\n")
                        st.write("\n")
                        save_name = f"{save_model}.pth"

                        #Grab the file from the directory to engage the web browser save dialouge, which is cleaner
                        with open(selected_model, 'rb') as f:
                            selected = f.read()

                        download_button_str = download_button(object_to_download=selected,
                                                            download_filename=save_name,
                                                            button_text=f'Save {save_model.replace("_", " ")}',
                                                            pickle_it=True)
                        st.markdown(download_button_str, unsafe_allow_html=True)
                    else:
                        st.error("Whoops. Something went wrong, did the model path change or a drive disconnect?")

    if model_settings_activity == "Model gallery":
        hdf5_file = None
        if hdf5_file != None:
            with h5py.File(hdf5_file, "r") as f:
                # List all groups
                image_names = list(f.keys())
            selections = st.multiselect("See model outputs for", image_names)
            st.write(selections)

            if st.button("View images"):
                with h5py.File(hdf5_file, "r") as data_file:
                    # List all groups
                    label_set = [Image.fromarray(np.uint8(np.array(data_file[key]["label"]))) for key in selections]
                    image_set = [Image.fromarray(np.uint8(np.array(data_file[key]["data"]))) for key in selections]
                    state.label_set = label_set
                    state.image_set = image_set

            control_col_1, control_col_2, control_col_3, control_col_4 = st.columns([1, 1, 1, 1])
            if state.image_set not in [None, "None"]:
                with control_col_1:
                    view_style = st.radio("View comparisons", ["Side by side", "On top of one another", "Overlay"])
                with control_col_2:
                    image_size = st.slider("Image size", min_value=1, max_value=1440, value=200)
            else:
                view_style = st.empty()
            if view_style == "Side by side":
                gallery_col_1, gallery_col_2, gallery_col_3 = st.columns([1, 1, 1])
                if state.image_set not in [None, "None"]:
                    with gallery_col_1:
                        st.image(state.image_set, width=image_size)

                    with gallery_col_2:
                        if state.label_set not in [None, "None"]:
                            st.image(state.label_set, width=image_size)

            if view_style == "On top of one another":
                if state.image_set not in [None, "None"]:
                    st.image(state.image_set, width=image_size)
                if state.label_set not in [None, "None"]:
                    st.image(state.label_set, width=image_size)

            if view_style == "Overlay":
                with control_col_3:
                    overlay_thresh = st.text_input("Overlay thresh level", 200)
                with control_col_4:
                    overlay_opacity = st.slider(label=f'Overlay opacity',
                                                min_value=0.0,
                                                max_value=1.0,
                                                value=0.5)
                    color = st.color_picker(label='Overlay Color', value='#ff1493')
                    rgb_color = ImageColor.getrgb(str(color))

                bottom_layer = [cv2.cvtColor(np.array(unseg).astype(np.uint8), cv2.COLOR_GRAY2BGR) for unseg in state.image_set]
                top_layer = [cv2.cvtColor(np.array(seg).astype(np.uint8), cv2.COLOR_GRAY2BGR) for seg in state.label_set]
                overlay_zip = zip(bottom_layer, top_layer)

                overlay_list = []
                for key, values in overlay_zip:
                    overlay = color_overlay(image=key, overlay_image=values,
                                            overlay_thresh=int(overlay_thresh),
                                            color=list(rgb_color),
                                            alpha=float(overlay_opacity),
                                            darkmode=False)
                    overlay_list.append(overlay)
                st.image(overlay_list, width=image_size)

            if st.checkbox("Select models list"):
                if st.checkbox("Define a different model directory"):
                    model_directory = st.text_input("Where is the model directory?")
                    model_list = glob.glob(str(pathlib.Path(model_directory).joinpath("*.pth")))
                    model_clean = [pathlib.Path(model).name for model in model_list]
                else:
                    model_directory = state.model_validation_directory
                    model_list = glob.glob(str(pathlib.Path(state.model_validation_directory).joinpath("*.pth")))
                    model_clean = [pathlib.Path(model).name for model in model_list]

                models_selected = st.multiselect("select the models to try:", model_clean)
                models_selected = [pathlib.Path(model_directory).joinpath(f"{try_model}") for try_model in models_selected]
                if st.button("Try em out!"):
                    all_results = []
                    for try_model in models_selected:
                        st.write(f"{try_model}")
                        net = model_initiation(model_path=str(try_model), cuda_index=state.use_gpu)
                        model_results = []
                        for image_name in state.image_set:
                            seg_image = three_class_segmentation(input_image=image_name, network=net)
                            model_results.append(seg_image)
                        all_results.append(model_results)
                    state.all_results = all_results
                if state.all_results not in [None, "None"]:
                    st.checkbox("View results")
                    for items in state.all_results:
                        st.image(items, width=image_size)






####
#
#   Start of app functions
#
####

def color_overlay(image, overlay_image, overlay_thresh=254, color=[100, 8, 58], alpha=0.5, darkmode=False):
    """
    Function to read in and overlay two black and white images in open cv2 format.
    :param image: An openCV style image array. Generally grey value data.
    :param overlay_image: An openCV style image array. Generally grey value data.
    :param overlay_thresh: Integer threshold value for the overlay.
    :param color: list RGB style color values for the overlay.
    :param alpha: float for the opacity
    :return: Return an openCV style image with a colored overlay.
    """
    img_out = image.copy()
    if darkmode == True:
        img_out[np.where((img_out == [255, 255, 255]).all(axis=2))] = [53, 51, 49]
    overlay_image = overlay_image.copy()
    ret, mask = cv2.threshold(src=overlay_image, thresh=int(overlay_thresh), maxval=255, type=cv2.THRESH_BINARY)
    mask[np.where((mask == [255, 255, 255]).all(axis=2))] = color
    img_out = cv2.addWeighted(src1=img_out, alpha=1, src2=mask, beta=alpha, gamma=0)
    return img_out


def model_initiation(model_path, cuda_index):
    net = UNet_Light_RDN(n_channels=8, n_classes=3)
    # Load in the trained model
    net.load_state_dict(torch.load(model_path, map_location=f'cuda:{int(cuda_index)}'))
    net.cuda()
    net.eval()
    return (net)

def three_class_segmentation(input_image, network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a SimpleITK readable format (e.g. "tif", "png", "jpg", "bmp", "mhd", "nii", etc.)
    :param input_image: A list of images to be segmented.
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :param network: The pytorch network to be used for the segmentation.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    net = network

    # Loop through the images in the folder and use the image name for the output name
    image = sitk.GetImageFromArray(input_image)
    #Check if the image is a vector and extract the first component, if so.
    if image.GetPixelID() == 13:
        image = sitk.VectorIndexSelectionCast(image, 0)

    #Rescale the image to 8 bit if it isn't already
    if image.GetPixelID() != 1:
        image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

    image = _setup_sitk_image(image, direction="z")
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).float() / 255.0
    image = image.cuda()

    # Turn all the gradients to false and get the maximum predictors from the network
    with torch.no_grad():
        pred = net(image)
    pred = pred.argmax(1)
    pred = pred.cpu().squeeze().data.numpy()

    seg_image = _view_predictor(pred=pred)
    return seg_image

def _setup_sitk_image(image_slice, direction="z"):
    """
    Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array

    direction = str(direction).lower()

    # Convert the image slice into a numpy array
    image = sitk.GetArrayFromImage(image_slice)

    # Deal with the variation in the 3d versus 2d array.
    if len(image.shape) == 2:
        if direction == "z":
            # Expand the z axis
            image = np.expand_dims(image, axis=2)
            # Check the dimensionality of the image, expand, transpose, for pytorch.
            image = image.transpose((2, 0, 1))
        elif direction == "y":
            image = np.expand_dims(image, axis=1)
            image = image.transpose((1, 0, 2))
        else:
            image = np.expand_dims(image, axis=0)
    return image


def _view_predictor(pred):
    """
    Internal function to convert predictions to an image for viewing in stremlit.
    """
    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_img = pred_img.astype(np.uint8)
    pred_img = Image.fromarray(pred_img, 'L')
    return pred_img

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    #https://gist.github.com/chad-m/6be98ed6cf1c4f17d09b7f6e5ca2978f
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    #Black button with deep pink 1 text
    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(0, 0, 0);
                color: rgb(255, 20, 147);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                font-size: 21px;
                font-family: 'Noto Sans', sans-serif;
                border-radius: 8px;
                border-width: 2px;
                border-style: solid;
                border-color: rgb(0, 0, 0);
                border-image: initial;
            }}  
            #{button_id}:hover {{
                border-color: #ff141d;
                color: rgb(230, 234, 241);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: #4c062c;
                color:  	      #f614ff;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

def _end_timer(start_timer, message : str =""):
    """
    Simple function to print the end of a timer in a single line instead of being repeated.
    :param start_timer: timer start called using timer() after importing: from time import time as timer.
    :param message: String that makes the timing of what event more clear (e.g. "segmenting", "meshing").
    :return: Returns a sring mesuraing the end of a timed event in seconds.
    """
    start = start_timer
    message = str(message)
    end = timer()
    elapsed = abs(start - end)
    if elapsed > 60:
        time_unit = ""
        time_formatted = _convert_seconds(seconds=elapsed)
    else:
        time_formatted = f"{float(elapsed):10.4f}"
        time_unit = "seconds."
    if message == "":
        st.text(f"Operation took: {time_formatted} {time_unit}")
    else:
        st.text(f"{message} took: {time_formatted} {time_unit}")

def _end_timer_sidebar(start_timer, message=""):
    """
    Simple function to print the end of a timer in a single line instead of being repeated.
    :param start_timer: timer start called using timer() after importing: from time import time as timer.
    :param message: String that makes the timing of what event more clear (e.g. "segmenting", "meshing").
    :return: Returns a sring mesuraing the end of a timed event in seconds.
    """
    start = start_timer
    message = str(message)
    end = timer()
    elapsed = abs(start - end)
    if elapsed > 60:
        time_unit = ""
        time_formatted = _convert_seconds(seconds=elapsed)
    else:
        time_formatted = f"{float(elapsed):10.4f}"
        time_unit = "seconds."
    if message == "":
        st.sidebar.text(f"Operation took: {time_formatted} {time_unit}")
    else:
        st.sidebar.text(f"{message} took: {time_formatted} {time_unit}")

def _convert_seconds(seconds: int) -> str:
    return str(timedelta(seconds=seconds)).rpartition(".")[0]

def open_windows_explorer(directory: Union[str, pathlib.Path], folder_designation):
    command_string = f"explorer {pathlib.Path(directory)}"
    with st.spinner(f"Opening {folder_designation} folder: {directory}"):
        subprocess.Popen(command_string, shell=True)


def generate_hdf5_streamlit(data_dir: Union[str, pathlib.Path], label_dir: Union[str, pathlib.Path], save_name: str):
    # read the sub file names from a file
    data_dir = pathlib.Path(data_dir)
    label_dir = pathlib.Path(label_dir)
    data_list = os.listdir(str(data_dir))
    label_list = os.listdir(str(label_dir))
    with st.spinner("Checking names..."):
        data_set_names = check_label_and_training_name(data_list=data_list, label_list=label_list)
    with st.spinner("Writing HDf5"):
        create_hdf5(save_name=save_name, data_set_names=data_set_names, data_dir=data_dir, label_dir=label_dir)

def generate_hdf5(data_dir: Union[str, pathlib.Path], label_dir: Union[str, pathlib.Path], save_name: str):
    # read the sub file names from a file
    data_dir = pathlib.Path(data_dir)
    label_dir = pathlib.Path(label_dir)
    data_list = os.listdir(str(data_dir))
    label_list = os.listdir(str(label_dir))
    data_set_names = check_label_and_training_name(data_list=data_list, label_list=label_list,to_streamlit=False)
    create_hdf5(save_name=save_name, data_set_names=data_set_names, data_dir=data_dir, label_dir=label_dir)


def check_label_and_training_name(data_list: List, label_list: List, to_streamlit=True) -> List:
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
            if to_streamlit:
                st.error(f"Unable to find a match for {label_name[idx]}. Please check if the cases, underscores, etc. match.")
            else:
                print(f"Unable to find a match for {label_name[idx]}. Please check if the cases, underscores, etc. match.")
    return data_set_names

def create_hdf5(save_name: str, data_set_names: List, data_dir: Union[str, pathlib.Path], label_dir: Union[str, pathlib.Path]):
    data_dir = pathlib.Path(data_dir)
    label_dir = pathlib.Path(label_dir)
    with h5py.File(save_name, "w") as image_file:
        # load train data to hdf5 file
        for data_name in data_set_names:
            sample_img = image_file.create_group(sep_string(data_name['label'], ".", n=1))
            sample_img.create_dataset('data', data=load_img(str(data_dir.joinpath(f"{data_name['data']}"))))
            sample_img.create_dataset('label', data=load_img(str(label_dir.joinpath(f"{data_name['label']}"))))


def generate_patches_streamlit(hdf5_file : Union[str, pathlib.Path], patches_csv: Union[str, pathlib.Path],
                               validation_csv: Union[str, pathlib.Path], train_ratio: float = 0.7,
                               stride: int = 32, output_size=256, always_train_csv: Union[str, bool] = False):

    with h5py.File(hdf5_file, 'r') as data_f:
        names_list = list(data_f.keys())
    # Set aside the labels you always want the model to be trained on ( e.g. fossils)
    if type(always_train_csv) is str:
        train_names, val_names = remove_from_validation_set(names_list=names_list,
                                                            always_train_csv=always_train_csv,
                                                            train_ratio=train_ratio)
    else:
        train_names, val_names = separate_names(names_list, train=train_ratio)
    st.info(f"Training model with {len(train_names)} and validating with {len(val_names)} images.")

    #Get patches and validation information for csv
    patches = get_patches(hdf5_file=hdf5_file, train_names=train_names, stride=stride, output_size=output_size)

    # May have to have a toggle to switch between.
    # This is likely not required. For experiments it should be the sk_shuffle so they are consistent.
    rand_shuffle(patches)

    #Get the information for the validation data
    with h5py.File(hdf5_file, 'r') as data_f:
        val = [[name, '0', '0', data_f[name]['label'][()].shape[0], data_f[name]['label'][()].shape[1]] for name in val_names]

    #Write patches and validation to a csv
    df_headers = ['name', 'top', 'left', 'h', 'w']
    patches = pd.DataFrame(patches)
    patches.columns = df_headers
    patches.to_csv(str(patches_csv), index=False)

    val = pd.DataFrame(val)
    val.columns = df_headers
    val.to_csv(str(validation_csv), index=False)

    st.info(f"Generated {len(patches)} patches")
    return val_names

def generate_patches(hdf5_file : Union[str, pathlib.Path], patches_csv: Union[str, pathlib.Path],
                               validation_csv: Union[str, pathlib.Path], train_ratio: float = 0.7,
                               stride: int = 32, output_size=256, always_train_csv: Union[str, bool] = False):

    with h5py.File(hdf5_file, 'r') as data_f:
        names_list = list(data_f.keys())
    # Set aside the labels you always want the model to be trained on ( e.g. fossils)
    if type(always_train_csv) is str:
        train_names, val_names = remove_from_validation_set(names_list=names_list,
                                                            always_train_csv=always_train_csv,
                                                            train_ratio=train_ratio, to_streamlit=False)
    else:
        train_names, val_names = separate_names(names_list, train=train_ratio)
    print(f"Entraînement avec {len(train_names)} images et validation avec {len(val_names)} images.")

    #Get patches and validation information for csv
    patches = get_patches(hdf5_file=hdf5_file, train_names=train_names, stride=stride, output_size=output_size)

    # May have to have a toggle to switch between.
    # This is likely not required. For experiments it should be the sk_shuffle so they are consistent.
    rand_shuffle(patches)

    #Get the information for the validation data
    with h5py.File(hdf5_file, 'r') as data_f:
        val = [[name, '0', '0', data_f[name]['label'][()].shape[0], data_f[name]['label'][()].shape[1]] for name in val_names]

    #Write patches and validation to a csv
    df_headers = ['name', 'top', 'left', 'h', 'w']
    patches = pd.DataFrame(patches)
    patches.columns = df_headers
    patches.to_csv(str(patches_csv), index=False)

    val = pd.DataFrame(val)
    val.columns = df_headers
    val.to_csv(str(validation_csv), index=False)

    print(f"Nombre de patches générés : {len(patches)} ")
    return train_names, val_names

def remove_from_validation_set(names_list: List, always_train_csv: Union[str, pathlib.Path], train_ratio: float, to_streamlit=True) -> Union[Tuple, Tuple]:
    always_train = pd.read_csv(str(pathlib.Path(always_train_csv)))
    if to_streamlit:
        st.write(f"Removing {len(always_train)} images from validation file.")
    else:
        print(f"Removing {len(always_train)} images from validation file.")
    #Stick the column into a list and then make sure we have unique variables with sets
    set_aside = always_train['Always_train'].to_list()
    names_list = list(set(names_list) - set(set_aside))
    names_list.sort()
    train_names, val_names = separate_names(names_list, train=train_ratio)
    train_names = train_names + set_aside
    return train_names, val_names

def get_patches(hdf5_file: Union[str, pathlib.Path], train_names: Union[List, Tuple], stride: int = 32, output_size: int = 256):
    patches = []
    with h5py.File(hdf5_file, 'r') as data_file:
        for name in train_names:
            shape = data_file[name]['label'][()].shape
            patches += slide_windows(name, shape, output_size=output_size, stride=stride)
        sk_shuffle(patches)
        # I wrote out the before and after and did a comparison and there are no differences.
        #This section appears to simply cast the first two items in patches as strings.
        # for idx in range(len(patches)):
        #     for j in range(len(patches[idx])):
        #         patches[idx][j] = str(patches[idx][j])
        return patches

def generate_ratios_streamlit(hdf5_file: Union[str, pathlib.Path], patches_csv: Union[str, pathlib.Path], class_num=3):
    patches = load_patches(patches_csv)
    st.info(f"Loaded {len(patches)} patches...")
    ratios = []
    with h5py.File(hdf5_file, 'r') as data_file:
        progress_bar = st.progress(0)
        img_count = 0
        for [name, top, left, h, w] in patches:
            mask = data_file[name]['label'][top: top+h, left: left+w]
            mask = dp.adjustMask(mask, class_num)

            size = 1.0
            for idx in range(len(mask.shape)):
                size *= mask.shape[idx]

            ratio = [(np.sum(mask == idx)/size) for idx in range(class_num)]
            ratios.append(ratio)
            img_count += 1
            iteration = np.floor((100 * img_count) / len(patches))
            progress_bar.progress(int(iteration))
    return ratios

def generate_ratios(hdf5_file: Union[str, pathlib.Path], patches_csv: Union[str, pathlib.Path], class_num=3):
    patches = load_patches(patches_csv)
    print(f"{len(patches)} patches chargés...")
    ratios = []
    with h5py.File(hdf5_file, 'r') as data_file:
        img_count = 0
        for [name, top, left, h, w] in patches:
            mask = data_file[name]['label'][top: top+h, left: left+w]
            mask = dp.adjustMask(mask, class_num)

            size = 1.0
            for idx in range(len(mask.shape)):
                size *= mask.shape[idx]

            ratio = [(np.sum(mask == idx)/size) for idx in range(class_num)]
            ratios.append(ratio)
    return ratios

def _setup_patches(patches_csv: Union[str, pathlib.Path],to_streamlit=True) -> pd.DataFrame:
    patches = pd.read_csv(str(patches_csv))
    if to_streamlit:
        st.info(f"Loaded {len(patches)} patches...")
    else:
        print(f"Loaded {len(patches)} patches...")
    return patches

def parallelize_ratios(df: pd.DataFrame, func: Callable, hdf5_file: Union[str, pathlib.Path],
                       class_num: int, n_cores: int = 4, to_streamlit=True) -> pd.DataFrame:
    #Modified from https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    #Get all but one core if it isn't specified for whatever reason.
    if type(n_cores) != int:
        n_cores = cpu_count() - 1
    if to_streamlit:
        st.info(f"Running process with {n_cores} cores.")
    else:
        print(f"Calcul avec {n_cores} coeur(s).")
    # Split the dataframe in subsections by the number of cores.
    df_split = np.array_split(df, n_cores)
    # Initiate the pool
    pool = Pool(n_cores)
    # Generate the parameters so they all can be mapped using starmap
    params = [(hdf5_file, df_part, class_num) for df_part in df_split]
    # Execute the function (in this case generate_ratio_multi) and then concat the results
    df = pd.concat(pool.starmap(func, params))
    pool.close()
    pool.join()
    return df

def generate_ratios_streamlit_multi(hdf5_file: Union[str, pathlib.Path], patches_csv: pd.DataFrame,
                                    class_num: int = 3) -> pd.DataFrame:
    patches = patches_csv
    ratios = []
    with h5py.File(hdf5_file, 'r') as data_file:
        for row in patches.itertuples():
            name = str(row.name)
            top = int(row.top)
            left = int(row.left)
            h = int(row.h)
            w = int(row.w)
            label_mask = data_file[name]['label'][top: top + h, left: left + w]
            label_mask = dp.adjustMask(label_mask, class_num)

            size = 1.0
            for idx in range(len(label_mask.shape)):
                size *= label_mask.shape[idx]

            ratio = [(np.sum(label_mask == idx)/size) for idx in range(class_num)]
            ratios.append(ratio)
    ratios = pd.DataFrame(ratios)
    ratios_header = [f"Class {num}" for num in range(class_num)]
    ratios.columns = ratios_header
    return ratios


def make_parallel(func):
    """
        Decorator used to decorate any function which needs to be parallized.
        After the input of the function should be a list in which each element is a instance of input fot the normal function.
        You can also pass in keyword arguements seperatley.
        https://medium.com/analytics-vidhya/python-decorator-to-parallelize-any-function-23e5036fb6a
        :param func: function
            The instance of the function that needs to be parallelized.
        :return: function
        : example:
        list_of_post_ids = list(range(1, 20))

        # Serial way of calling the function
        results = []
        for post_id in list_of_post_ids:
        res = sample_function(post_id)
        results.append(res)

        # Paralleized way of calling the function
        results = make_parallel(sample_function)(list_of_post_ids)
    """

    @wraps(func)
    def wrapper(lst):
        """

        :param lst:
            The inputs of the function in a list.
        :return:
        """
        # the number of threads that can be max-spawned.
        # If the number of threads are too high, then the overhead of creating the threads will be significant.
        # Here we are choosing the number of CPUs available in the system and then multiplying it with a constant.
        # In my system, i have a total of 8 CPUs so i will be generating a maximum of 16 threads in my system.
        number_of_threads_multiple = 2 # You can change this multiple according to you requirement
        number_of_workers = int(os.cpu_count() * number_of_threads_multiple)
        if len(lst) < number_of_workers:
            # If the length of the list is low, we would only require those many number of threads.
            # Here we are avoiding creating unnecessary threads
            number_of_workers = len(lst)

        if number_of_workers:
            if number_of_workers == 1:
                # If the length of the list that needs to be parallelized is 1, there is no point in
                # parallelizing the function.
                # So we run it serially.
                result = [func(lst[0])]
            else:
                # Core Code, where we are creating max number of threads and running the decorated function in parallel.
                result = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executer:
                    bag = {executer.submit(func, i): i for i in lst}
                    for future in concurrent.futures.as_completed(bag):
                        result.append(future.result())
        else:
            result = []
        return result
    return wrapper

def download_model(model, model_name):
    """

    : param model:
    : param model_name:
    : return:
    """
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download={model_name}>Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def read_train_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as open_file:
        train_yaml_file = yaml.load(open_file.read(), Loader=yaml.FullLoader)
    return train_yaml_file

def read_test_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as open_file:
        test_yaml_file = yaml.load(open_file.read(), Loader=yaml.FullLoader)
    return test_yaml_file

def _custom_logo(image_path, image_text):
    st.sidebar.markdown(
        """<style>
        figcaption {
                   font-weight:100 !important;
                   font-size:27.5px !important;
                   text-align: left !important;
                   color: black !important;
                   z-index: 1;             
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"""<figure>
                                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="my img"  width="300"/>
                                <figcaption>{image_text}</figcaption>
                            </figure>""",
                        unsafe_allow_html=True)

def rescale_label_proper(input_image, input_file_name: str):
    nda = sitk.GetArrayFromImage(input_image)
    rescale = np.where(nda == 1, 128, nda)
    rescale = np.where(rescale == 2, 255, rescale)
    rescaled = sitk.GetImageFromArray(rescale)
    rescaled = sitk.Cast(rescaled, sitk.sitkUInt8)
    rescaled.CopyInformation(input_image)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(f"{str(input_file_name)}")
    writer.Execute(rescaled)

def rescale_intensity(inputFilename: str, writeOut: bool=True, file_type: str="", outDir: Union[str, pathlib.Path]=""):
    """
    Load in a 2d image file and rescale for data augmentation.
    :param inputFilename: Name of file to be rescaled. Can be anything that SimpleITK reads.
    :return: Returns a rescaled tif image.
    @rtype: object
    """

    out_name = str(inputFilename).rsplit(".", 1)[0]
    if file_type == "":
        file_type = str(inputFilename).rsplit(".", 1)[-1]
    if "." in str(file_type):
        file_type.replace(".", "")
    if "\\" in out_name or "/" in out_name:
        out_name = pathlib.Path(out_name).parts[-1]



    inputImage = sitk.ReadImage(str(inputFilename), sitk.sitkUInt8)
    MinMax = sitk.MinimumMaximumImageFilter()
    try:
        MinMax.Execute(inputImage)
    except RuntimeError:
        if inputImage.GetNumberOfComponentsPerPixel == 3:
            inputImage_array = sitk.GetArrayFromImage(inputImage)
            inputImage_array = rgb_to_gray(inputImage_array)
            inputImage_array = sitk.GetImageFromArray(inputImage_array)

            #TODO make function for this
            #Grab the relevant metadata to set
            res = inputImage.GetSpacing()
            direction = inputImage.GetDirection()
            origin = inputImage.GetOrigin()

            #Set the metadata
            inputImage_array.SetOrigin(origin)
            inputImage_array.SetDirection(direction)
            inputImage_array.SetSpacing(res)
        else:
            #place holder until I figure out what to do
            print("Input Image type not understood. ")

    #Rescale to prevent overflow
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    rescaleFilt = sitk.RescaleIntensityImageFilter()
    rescaleFilt.SetOutputMinimum(0)
    imageMax = MinMax.GetMaximum()
    if imageMax == 255.0:
        rescaleFilt.SetOutputMaximum(192)
    elif imageMax < 255.0 and imageMax > 192.0:
        rescaleFilt.SetOutputMaximum(170)
    else:
        rescaleFilt.SetOutputMaximum(255)
    rescaled = rescaleFilt.Execute(inputImage)
    rescaled = sitk.Cast(rescaled, sitk.sitkUInt8)
    if writeOut == True:
        writer = sitk.ImageFileWriter()
        if outDir != "":
            out_name = pathlib.Path(outDir).joinpath(out_name)
        writer.SetFileName(f"{str(out_name)}_rescaled.{file_type}")
        writer.Execute(rescaled)
    else:
        return rescaled

def downscale_intensity(inputFilename, downscale_value=60, writeOut=True, file_type="", outDir=""): # downscale value
    """
    Load in a 2d image file and rescale for data augmentation.
    :param inputFilename: Name of file to be resclaed. Can be anything that SimpleITK reads.
    :return: Returns a rescaled tif image.
    """

    out_name = str(inputFilename).rsplit(".", 1)[0]
    if file_type == "":
        file_type = str(inputFilename).rsplit(".", 1)[-1]
    if "." in str(file_type):
        file_type.replace(".", "")
    if "\\" in out_name or "/" in out_name:
        out_name = pathlib.Path(out_name).parts[-1]


    inputImage = sitk.ReadImage(str(inputFilename), sitk.sitkUInt8)
    MinMax = sitk.MinimumMaximumImageFilter()
    try:
        MinMax.Execute(inputImage)
    except RuntimeError:
        if inputImage.GetNumberOfComponentsPerPixel == 3:
            inputImage_array = sitk.GetArrayFromImage(inputImage)
            inputImage_array = rgb_to_gray(inputImage_array)
            inputImage_array = sitk.GetImageFromArray(inputImage_array)

            #TODO make function for this
            #Grab the relevant metadata to set
            res = inputImage.GetSpacing()
            direction = inputImage.GetDirection()
            origin = inputImage.GetOrigin()

            #Set the metadata
            inputImage_array.SetOrigin(origin)
            inputImage_array.SetDirection(direction)
            inputImage_array.SetSpacing(res)
        else:
            #place holder until I figure out what to do
            print("Input Image type not understood. ")

    #Rescale to prevent overflow
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    rescaleFilt = sitk.RescaleIntensityImageFilter()
    rescaleFilt.SetOutputMinimum(0)
    imageMax = MinMax.GetMaximum()
    if imageMax <= float(downscale_value):
        pass
    else:
        rescaleFilt.SetOutputMaximum(downscale_value)
        rescaled = rescaleFilt.Execute(inputImage)
        rescaled = sitk.Cast(rescaled, sitk.sitkUInt8)
        if writeOut == True:
            writer = sitk.ImageFileWriter()
            if outDir != "":
                out_name = pathlib.Path(outDir).joinpath(out_name)
            writer.SetFileName(f"{str(out_name)}_downscaled.{file_type}")
            writer.Execute(rescaled)
        else:
            return rescaled


def glob_flat_list(search_directory: str, file_types: Union[str, List[str]], unique: bool= False) -> List:
    search_directory = pathlib.Path(search_directory)
    file_types = list(file_types)
    new_images = [glob.glob(str(search_directory.joinpath(f"*.{types}"))) for types in file_types]
    flat_list = [item for sublist in new_images for item in sublist]
    if unique:
        flat_list = list(set(flat_list))
    flat_list.sort()
    return flat_list


def file_selector(folder_path='.', extension="", selectbox_text="", unique_key=""):
    if folder_path == '.' or '':
        folder_path = pathlib.Path.cwd()
    filenames = os.listdir(folder_path)
    filenames.sort(reverse=True)
    if extension != "":
        filenames = [num for num in filenames if extension in num]
    if unique_key == "":
        selected_filename = st.selectbox(f'{selectbox_text}', filenames)
    else:
        selected_filename = st.selectbox(f'{selectbox_text}', filenames, key=unique_key)
    try:
        return os.path.join(folder_path, selected_filename)
    except:
        pass


def clean_image(inputFilename, suffix="", out_name="", out_type="", out_dir="", remove_space=True, remove_dblundr=True, remove_hyphen=True, to_streamlit=False):
    """
    Function to read in and write out a 2d image. Useful for file type conversion and renaming.
    :param inputFilename: Name of file to be resclaed. Can be anything that SimpleITK reads.
    :param suffix: string that will appear before the out_type
    :return: Returns a 2d image.
    """
    if out_name == "":
        out_name = inputFilename.rsplit(".", 1)[0]
    else:
        out_name = inputFilename

    if out_name == "" and out_type == "":
        out_name = inputFilename.rsplit(".", 1)[0]
        out_type = inputFilename.rsplit(".", 1)[-1]
    elif out_name == "" and out_type != "":
        out_name = inputFilename.rsplit(".", 1)[0]
        out_type = out_type
    elif out_name != "" and out_type == "":
        out_name = out_name
        out_type = inputFilename.rsplit(".", 1)[-1]
    else:
        out_name = str(out_name)
        out_type = str(out_type)

    if "." in out_type:
        out_type.replace(".", "")

    if out_dir != "":
        if "\\" in out_name or "/" in out_name:
            out_name = pathlib.Path(out_name).parts[-1]
            if remove_space == True:
                out_name = out_name.replace(" ", "")
            if remove_dblundr == True:
                out_name = out_name.replace("__", "_")
            if remove_hyphen == True:
                out_name = out_name.replace("-", "_")


        out_name = str(pathlib.Path(out_dir).joinpath(out_name))

    inputImage = sitk.ReadImage(inputFilename)
    if inputImage.GetPixelID() != 1:
        inputImage = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)
    if to_streamlit:
        st.write((f"\nWriting out {out_name} as {out_type}....\n"))
    #else:
        #print(f"\nWriting out {out_name} as {out_type}....\n")
    writer = sitk.ImageFileWriter()
    writer.SetFileName(f"{out_name}{suffix}.{out_type}")
    writer.Execute(inputImage)


def _check_label(inputImage: sitk.Image, expected_classes: int, to_streamlit=True):
    nda = sitk.GetArrayFromImage(inputImage)
    num_classes, class_proportion = np.unique(nda, return_counts=True)
    if to_streamlit:
        print(num_classes)
    if len(num_classes) != expected_classes:
        if to_streamlit:
            st.write(f"There are {num_classes} instead of {expected_classes} classes in the label!")
            st.write(f"{class_proportion}")
        return None
    elif np.array_equal(num_classes, np.array([0, 1, 2])):
        return False
    else:
        return True

def check_for_match(missing_file: str, check_filelist: List, to_streamlit=True):
    cases = [(missing_file, check_file) for check_file in check_filelist]
    matches = []
    for checking, file_name in cases:
        similarity = 0
        too_many = 0
        too_few = 0
        for i, s in enumerate(difflib.ndiff(checking, file_name)):
            if s[0] == ' ':
                similarity += 1
            elif s[0] == '-':
                too_many += 1
            elif s[0] == '+':
                too_few += 1
        sim_score = similarity - (too_few + too_many)
        if sim_score > 0:
            matches.append(file_name)
    if len(matches) == 0:
        matches = difflib.get_close_matches(word=checking, possibilities=cases, n=1, cutoff=0.2)
    if len(matches) != 0:
        pairs = [(missing_file, matched) for matched in matches]
        for checking, match_file in pairs:
            add_text = difflib.Differ().compare(checking, match_file)
            minus_text = difflib.Differ().compare(checking, match_file)
            add_text = [''.join(x).replace("+ ", "") for x in add_text if '+ ' in x]
            minus_text = [''.join(x).replace("- ", "") for x in minus_text if '- ' in x]
            if to_streamlit:
                st.write(f"\nDifferences between {checking} and {match_file}:")
            else:
                print(f"\nDifferences entre {checking} et {match_file}:")
            if len(add_text) > 0:
                if len(add_text) == 1:
                    if to_streamlit:
                        st.write(f"{len(add_text)} character needs to be added to match {checking}:")
                    else:
                        print(f"{len(add_text)} caractère(s) doit être ajouté pour correspondre à  {checking}:")
                else:
                    if to_streamlit:
                        st.write(f"{len(add_text)} character needs to be added to match {checking}:")
                    else:
                        print(f"{len(add_text)} caractère(s) doit être ajouté pour correspondre à  {checking}:")
                too_much_text = str(", ".join(add_text).replace(",", ""))
                color_print(too_much_text, color="yellow")

            if len(minus_text) > 0:
                if len(minus_text) == 1:
                    if to_streamlit:
                        st.write(f"{len(minus_text)} character must be changed or removed to match {checking}:")
                    else:
                        print(f"{len(add_text)} caractère(s) doit être changés ou supprimés pour correspondre à  {checking}:")
                else:
                    if to_streamlit:
                        st.write(f"{len(minus_text)} character must be changed or removed to match {checking}:")
                    else:
                        print(f"{len(add_text)} caractère(s) doit être changés ou supprimés pour correspondre à  {checking}:")
                too_little_text = str(", ".join(minus_text).replace(",", ""))
                color_print(text=too_little_text, color="blue")
            if to_streamlit:
                st.markdown(f"")
                st.markdown(f"")
        return matches
    else:
        if to_streamlit:
            st.warning(f"No good matches for {checking} found :frowning: (Sometimes happens with short file names)")
            st.markdown(f"")
            st.markdown(f"")
        else:
            st.warning(f"Pas de bonne(s) correspondance(s) trouvée(s) pour {checking} (Cela arrive parfois avec des noms de fichiers courts)")


def color_print(text: str, color: str):
    st.markdown(f'<font color={color}>{text}</font>', unsafe_allow_html=True)


def subtract_images(inputImage1: sitk.Image, inputImage2: sitk.Image):
    """
    Function to subtract two SimpleITK images using the Subtract filter.

    :param inputImage1: A SimpleITK image.
    :param inputImage2: A SimpleITK image.
    :return: Returns a single SimpleITK image.
    """
    start = timer()

    # Subtract the two images together
    #print("Subtracting...")
    subtracted = sitk.Subtract(inputImage1, inputImage2)
    #_end_timer(start, message="Subtracting the two images")
    return subtracted


def save_state_values(state, user):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(f"{user}_RDN_saved_state.json")
    session_dict = {"model_path": [str(state.model_path)],
                    "model": [str(state.model)],
                    "input_path": [str(state.input_path)],
                    "input_type": [state.input_type],
                    "output_path": [str(state.output_path)],
                    "out_type": [str(state.out_type)],
                    "parm_file": [str(state.parm_file)]}
    session_dict = pd.DataFrame.from_dict(session_dict)
    session_dict.to_json(saved_dir)
    st.write(f"Saved settings for {user}!")

def initiate_cuda():
    num_gpus = torch.cuda.device_count()
    #st.write(num_gpus)
    if num_gpus == 0:
        return False
    elif num_gpus == 1:
        return 0
    else:
        return "multiple"

def setup_gpu(device_num, state):
    if device_num == 0:
        state.use_gpu = 0
    elif device_num in [False, None]:
        st.error("GPU not found!")
    elif device_num == "multiple":
        with st.expander("Select gpu"):
            st.write("Multiple gpu's found")
            st.write("Look at you with the sweet setup... :smile:")
            num_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_gpus))
            device_num = st.selectbox('Select the gpu you want to use to train', gpu_list)
            state.use_gpu = device_num
    else:
        state.use_gpu = device_num
    try:
        torch.cuda.set_device(state.use_gpu)
    except ValueError:
        if st.write(torch.cuda.is_available()):
            state.use_gpu = 0
            torch.cuda.set_device(state.use_gpu)
    return state.use_gpu


def _convert_size(sizeBytes):
    """
    Function to return file size in a human readable manner.
    :param sizeBytes: bytes calculated with file_size
    :return:
    """
    if sizeBytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(sizeBytes, 1024)))
    p = math.pow(1024, i)
    s = round(sizeBytes / p, 2)
    return "%s %s" % (s, size_name[i]), s


####
# Changing anything below here will break everything
###



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_script_run_ctx().session_id
    session_info = runtime.get_instance()._session_mgr.get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()