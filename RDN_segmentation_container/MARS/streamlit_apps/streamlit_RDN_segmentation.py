"""
Streamlit app to utilize the RDN segmentation Yazdani et al., 2020 2019 54th Asilomar Conference on Signals, Systems,
and Computers.

GUI and additional functionality written by NB Stephens (github.com/NBStephens) nbs49@psu.edu

If utilizing the meshes created in here please be sure to cite to wonderful projects:

https://simpleitk.org/
https://github.com/SimpleITK

PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)
https://joss.theoj.org/papers/10.21105/joss.01450
https://github.com/pyvista/pyvista

# Check pytorch env
python -m torch.utils.collect_env
#TODO fix first run issues with input/output path, thinks previous input is nan
#TODO Make it so the generated par file can be in table format
"""


import os
import io
import cv2
import sys
import vtk
import math
import time
import glob
import torch
import base64
import socket
import pathlib
import builtins
import tempfile
import subprocess
import multiprocessing
import numpy as np
import pyvista as pv
import SimpleITK as sitkb
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from PIL import Image, ImageColor
from timeit import default_timer as timer




# TODO only dark mode
# TODO make portion for comparing model outputs
# TODO make it open the mesh
# TODO paraview launch
# TODO cloudcompare
# TODO insert naming scheme parameter
# TODO option to put a suffix
# TODO add to load a slice and try it out
# TODO fix the mesher warnings and fail log

# Reads where this script is launched from so you can import all the other functionality
script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir.parent.parent))
 
from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import *
from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import (
    _setup_image,
    _return_predictors,
    _get_threads,
    _get_outDir,
    _get_inDir,
)
from MARS.streamlit_apps.net.unet_light_rdn import (
    UNet_Light_RDN,
)

import streamlit as st
from streamlit.runtime.legacy_caching.hashing import _CodeHasher
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from streamlit.web.server import Server
from streamlit import runtime




# Variables that get reused on the various pages
# TODO add tiff in input list
# TODO delete your user information and then run it from start to catch bugs
supported_file_types = ["mhd", "nii", "tif", "png", "jpg", "bmp", "dcm"]
slice_types = ["tif", "png", "jpg", "bmp", "dcm"]
volume_types = ["mhd", "mha", "nii", "vtk"]

# Get the small logo for the tab
script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
tab_logo = Image.open(str(script_dir.joinpath("moon_square.png")))

# This is a beta feature to control the default elements of the app
try:
    st.set_page_config(
        page_title="RDN Segmentation",
        page_icon=tab_logo,
        layout="wide",
        initial_sidebar_state="auto",
    )
except AttributeError:
    st.set_page_config(
        page_title="RDN Segmentation",
        page_icon=tab_logo,
        layout="wide",
        initial_sidebar_state="auto",
    )


# Defines the body of the gui
def main():
    """
    Streamlit GUI to perform segmentation of grey value images using a trained RDN 3-class segmentation network
    implemented in pytorch. Reading and writing of values is predominantly handled with SimpleITK.
    """
    current_user = _get_user()

    # Load in the logo
    _load_MARS_logo()

    # Defines the pages on the side bar and what they show at the top
    pages = {
        "Settings": page_settings,
        "Single Segmentation": page_segmentations,
        "Batch segmentation": page_batch_segmentations,
        "View segmentations": page_midslice_viewer,
        "Batch meshing": page_batch_meshing,
    }
   
    # Saves settings between pages.
    state = _get_state()

    page = st.sidebar.radio("Navigation:", tuple(pages.keys()))

    ###
    st.sidebar.subheader("Common operations")

    # Button to save state values
    if st.sidebar.button("Save settings"):
        save_state_values(state=state, user=current_user)

    # Button to update graphics card information
    if st.sidebar.button("Initialize GPU"):
        initiate_cuda(state)

    # Launch notepad++ on windows.
    if st.sidebar.button("Parameter notepad++"):
        if state.parm_file:
            parm_file_loc = pathlib.Path(state.parm_file).as_posix()
            subprocess.run(f'start notepad++ "{str(parm_file_loc)}"', shell=True)
        else:
            st.warning("No parameter file set, opening notepad++ instead")
            subprocess.run(f"start notepad++", shell=True)

    if st.sidebar.button("Parameter Sublime Text"):
        sublime_dir = pathlib.Path(r"C:\Program Files\Sublime Text 3\subl.exe")
        if state.parm_file:
            parm_file_loc = pathlib.Path(state.parm_file).as_posix()
            command = subprocess.run(
                f'"{str(sublime_dir)}" "{str(parm_file_loc)}"', shell=True
            )
            exit_status = str(command).rsplit("=", 1)[1]
            if exit_status == "1)":
                st.sidebar.error(
                    "Whoops, it looks like Sublime isn't installed in the expected location!"
                )

        else:
            st.warning("No parameter file set, opening notepad++ instead")
            subprocess.run(f'"{str(sublime_dir)}"', shell=True)

    # Button to clear GPU memory
    if st.sidebar.button("Clear GPU"):
        torch.cuda.empty_cache()
        st.write("GPU memory cleared.")

    # Button to clear state values
    if st.sidebar.button("Clear settings"):
        state.clear()
        st.write("Settings cleared.")

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def page_settings(state):
    """
    Page to define the settings for the segmentation. If using batch mode, a parameter file is needed.

    """
    current_user = _get_user()

    # The supported file types to read and write.
    file_types = supported_file_types

    # Writes a title for the settings
    st.title(":wrench: Settings")

    # Help file
    # with st.sidebar.expander("HELP!", expanded=True):
    #    st.info("Note: Settings may be saved between runs by clicking 'Save settings', and restored by checking"
    #                    " the appropriate boxes.")
    #    st.markdown("**Steps:**")
    #    st.markdown("1) Initialize the GPU.")
    #    st.markdown("2) Paste the model folder location and select the desired .pth file.")
    #    st.markdown("*For single volume or 2d slice segmentations:*")
    #    st.markdown("2) Define the input location and file type, as well as the output location and file type")
    #    st.markdown("*For multiple segmentations check 'batch_mode'*")
    #    st.markdown("3) Paste the location of the parameter file and select the csv")

    # Writes and info cell with the graphics card infromation.
    if state.cuda_mem == None:
        st.error("GPU not initialized!")
    else:
        st.write(
            f"Segmenting with {torch.cuda.get_device_name(state.use_gpu)}, {state.cuda_mem} of memory."
        )

    col1, col2 = st.columns(2)
    # Writes out a thing line across the gui page.
    with col1:
        st.write("---")
        st.subheader("Model settings")

        # Check box to read previous state values. Makes it easy to use the same models
        model_radio_list = [
            "Previous model",
            "Only previous path",
            "New model settings",
        ]
        query_params = st.experimental_get_query_params()

        # Query parameters are returned as a list to support multiselect.
        # Get the first item in the list if the query parameter exists.
        default = int(query_params["activity"][0]) if "activity" in query_params else 0
        model_settings_activity = st.radio(
            "Define the model location:", options=model_radio_list, index=0
        )

        if model_settings_activity:
            st.experimental_set_query_params(
                activity=model_radio_list.index(model_settings_activity)
            )

            if model_settings_activity == "Previous model":
                if st.button("Load previous model settings"):
                    restored_state = load_state_values(state, user=current_user)
                    state.model_path = restored_state.model_path
                    st.write(state.model_path)
                    state.model = pathlib.Path(restored_state.model)
                    st.write(pathlib.Path(state.model).parts[-1])

            if model_settings_activity == "Only previous path":
                restored_state = load_state_values(state, user=current_user)
                state.model_path = restored_state.model_path
                state.model = file_selector(state.model_path, extension="pth")
                save_state_values(state, current_user)

            if model_settings_activity == "New model settings":
                state.model_path = pathlib.Path(st.text_input("Model location", ""))
                
                state.model = file_selector(state.model_path, extension="pth")
                save_state_values(state, current_user)
                

    with col2:
        st.write("---")
        st.subheader("Batch mode Input/Output settings")

        # Check box to read previous state values. Makes it easy to use the same models
        parm_radio_list = [
            "Previous parameters",
            "Load from new location",
            "Build parameters from directory",
            "Parameters from Medtool par file",
        ]
        parm_query_params = st.experimental_get_query_params()

        # Query parameters are returned as a list to support multiselect.
        # Get the first item in the list if the query parameter exists.
        parm_default = (
            int(parm_query_params["activity"][0])
            if "activity" in parm_query_params
            else 0
        )
        parm_settings_activity = st.radio(
            "Paste in the parameter file location:", parm_radio_list, index=parm_default
        )
        if parm_settings_activity == "Previous parameters":
            # If segmenting in batch you can set it the location here.
            restored_state = load_state_values(state, user=current_user)
            if str(restored_state.parm_file) != "None" or "nan":
                state.existing_parm = True

        if parm_settings_activity == "Load from new location":
            state.existing_parm = "defining"
            parm_file_loc = pathlib.Path(st.text_input("Parameter file location", ""))
            state.parm_file = file_selector(parm_file_loc, extension=".csv")
            if st.checkbox("Load and view parameter file", key="99990287"):
                loaded_parm = read_parameter_file(state.parm_file)
                st.write(loaded_parm)
                save_state_values(state=state, user=current_user)

        if parm_settings_activity == "Build parameters from directory":
            state.existing_parm = "defining"
            in_dir_loc = pathlib.Path(
                st.text_input("Segmentation input directory ", "")
            )
            in_file_type = st.selectbox("Pick input file type:", file_types)
            out_dir_loc = pathlib.Path(
                st.text_input("Segmentation output directory ", "")
            )
            out_file_type = st.selectbox("Pick output file type:", file_types)
            if st.checkbox("Generate parameter file"):
                new_parm_file = parm_from_directory(
                    input_dir=in_dir_loc,
                    input_file_type=in_file_type,
                    output_dir=out_dir_loc,
                    output_file_type=out_file_type,
                )
                st.write(new_parm_file)
                if st.checkbox("Save parameter file"):
                    state.existing_parm = "defining"
                    parm_file_output = pathlib.Path(st.text_input("Save directory", ""))
                    parm_file_name = st.text_input(
                        "Save name",
                    )
                    if st.button("Save parameter!"):
                        parm_outname = parm_file_output.joinpath(
                            f"{parm_file_name}.csv"
                        )
                        new_parm_file.to_csv(parm_outname)
                        st.info(f"Parameter file saved as {parm_file_output} :smile:")
                        state.parm_file = parm_outname
                        save_state_values(state=state, user=current_user)

        if parm_settings_activity == "Parameters from Medtool par file":
            in_dir_loc = pathlib.Path(st.text_input("Medtool par input directory ", ""))
            in_file_type = st.selectbox("Image input file types:", file_types)
            medtool_par_file = file_selector(in_dir_loc, extension=".par")
            out_dir_loc = pathlib.Path(
                st.text_input("RDN par file output directory ", "")
            )

            if st.checkbox("Convert parameter file"):
                RDN_parm_file = parm_from_par(
                    par_file=medtool_par_file, out_type="mhd", input_type=in_file_type
                )
                st.write(RDN_parm_file)

            if st.button("Save converted parameter file"):
                RDN_output = out_dir_loc.joinpath(f"{medtool_par_file[:-4]}_RDN.csv")
                RDN_parm_file.to_csv(RDN_output, sep=",")
                state.parm_file = str(RDN_output)
                save_state_values(state=state, user=current_user)
                
    with st.expander("Check parameter file", False):
        if state.existing_parm == "defining":
            pass
        elif state.existing_parm:
            if st.checkbox("Load and view previous parameter file", False):
                restored_state = load_state_values(
                    state, user=current_user, verbose=False
                )
                parm_file_loc = pathlib.Path(restored_state.parm_file)
                state.parm_file = parm_file_loc
                if str(state.parm_file) == "None":
                    st.warning(
                        "Couldn't find previous parameters file, please choose location."
                    )
                else:
                    st.info(f"Loaded parameter file: {state.parm_file}")
                    loaded_parm = read_parameter_file(state.parm_file)
                    st.write(loaded_parm)
                    save_state_values(state=state, user=current_user)
        else:
            st.error("Couldn't find previous parameters file!")

    st.write("---")

    # If only segmenting a bunch of 2d images or a single volume you can define that here.
    # Write out whatever state settings are defined in the display_state_values
    st.title("Current settings")
    display_state_values(state)


# Rescale to prevent overflow
def page_segmentations(state):
 
    """
    Sets up the single segmentation page
    """
    current_user = _get_user()
    state.twoD_to_threeD = False

    if len(state._state["data"]) == 0:
        st.error("Must fill out settings!")

    st.title("Single segmentation settings:")

    single_radio_list = [
        "Use previous input/output settings",
        "New input/output settings",
    ]

    single_settings_activity = st.experimental_get_query_params()
    st.write("---")
    # Query parameters are returned as a list to support multiselect.
    # Get the first item in the list if the query parameter exists.
    parm_default = (
        int(single_settings_activity["activity"][0])
        if "activity" in single_radio_list
        else 0
    )
    single_settings_activity = st.radio(
        "Select input output settings?", single_radio_list, index=parm_default
    )

    if single_settings_activity == "Use previous input/output settings":
        restored_state = load_state_values(state, user=current_user)

        if st.checkbox("Previous input path", value=True, key="9990363"):
            if is_state_value_empty(restored_state.input_path, verbose=False):
                st.warning("No saved path found, please enter a new location")
                state.output_path = pathlib.Path(st.text_input("Output location", ""))
                save_state_values(state=state, user=current_user)
            else:
                state.input_path = pathlib.Path(restored_state.input_path)
                st.write(state.input_path)
                save_state_values(state=state, user=current_user)
        else:
            state.input_path = pathlib.Path(st.text_input("Input location", ""))
            save_state_values(state=state, user=current_user)

        if st.checkbox("Previous input type", value=True, key="9990376"):
            state.input_type = restored_state.input_type
            st.write(state.input_type)
            save_state_values(state=state, user=current_user)
        else:
            try:
                state.input_type = st.selectbox(
                    "Input file types",
                    supported_file_types,
                    supported_file_types.index(state.input_type)
                    if state.input_type
                    else 0,
                )
            except ValueError:
                state.input_type = "mhd"
            save_state_values(state=state, user=current_user)

        if st.checkbox("Previous output path", value=True, key="9990383"):
            if is_state_value_empty(restored_state.output_path, verbose=False):
                st.warning("No saved path found, please enter a new location")
                state.output_path = pathlib.Path(
                    st.text_input("Output location", "", key="9990393")
                )
                save_state_values(state=state, user=current_user)
            else:
                st.write(state.output_path)
                save_state_values(state=state, user=current_user)
        else:
            state.output_path = pathlib.Path(
                st.text_input("Output location", "", key="9990420")
            )
            save_state_values(state=state, user=current_user)

        if st.checkbox("Previous output type", value=True):
            state.out_type = restored_state.out_type
            st.write(state.out_type)
            save_state_values(state=state, user=current_user)
        else:
            try:
                state.out_type = st.selectbox(
                    "Select value.",
                    supported_file_types,
                    supported_file_types.index(state.out_type) if state.out_type else 0,
                )
            except ValueError:
                state.out_type = "mhd"
            save_state_values(state=state, user=current_user)

    if single_settings_activity == "New input/output settings":
        state.input_path = pathlib.Path(st.text_input("Input location", ""))
        if str(state.input_type) == "nan":
            state.input_type = st.selectbox(
                "Input file type", supported_file_types, key="99990420"
            )
        else:
            st.write(state.input_type)
            state.input_type = st.selectbox(
                "Input file type",
                supported_file_types,
                supported_file_types.index(state.input_type) if state.input_type else 0,
            )
        state.output_path = pathlib.Path(st.text_input("Output location", ""))

        if str(state.out_type) == str(None):
            state.out_type = st.selectbox("Output file type", supported_file_types)
        else:
            state.out_type = st.selectbox(
                "Output file type",
                supported_file_types,
                supported_file_types.index(state.out_type) if state.out_type else 0,
            )
        save_state_values(state=state, user=current_user)

    if state.input_type in volume_types:
        st.write(f"Please select {state.input_type} file:")
        individual_seg = file_selector(
            folder_path=str(state.input_path), extension=f"{state.input_type}"
        )
        st.info(f"Selected {individual_seg} for segmentation...")
    elif state.input_type in slice_types:
        search_dir = pathlib.Path(state.input_path)
        individual_seg_list = glob.glob(
            str(search_dir.joinpath(f"*.{state.input_type}"))
        )
        st.write(f"Found {len(individual_seg_list)} {state.input_type} files")
        if st.checkbox("Show image file list"):
            st.write(individual_seg_list)
    else:
        st.warning("Must indicate and input file type and make a file selection!")

    if st.checkbox("Rescale to 255 prior to segmentation?"):
        state.rescale_prior = True
        if st.checkbox("Check for metal/bright inclusions?"):
            if st.checkbox("View mid-plane histogram?", key="9990438"):
                if state.input_type in slice_types:
                    st.error(
                        "Histograms are only supported with volume types at this time."
                    )
                    st.info(
                        "Please email the author at nbs49@psu.edu if you want me emphasize adding this."
                    )
                else:
                    input_path = pathlib.Path(state.input_path)
                    input_vol = input_path.joinpath(individual_seg)
                    if st.checkbox("Log scale", key="9990445"):
                        get_midplane_histogram(input_vol, log=True)
                    else:
                        get_midplane_histogram(input_vol, log=False)

            state.check_for_metal = True
            state.cut_off = st.text_input("Threshold out values above:", 200)
    else:
        state.rescale_prior = False
        state.check_for_metal = False

    st.write("---")
    in_check = str(state.input_type)
    out_check = str(state.out_type)

    _check_for_slice_to_vol(
        state,
        input_type=in_check,
        out_type=out_check,
        slice_types=slice_types,
        volume_types=volume_types,
        verbose=True,
    )

    if not state.output_path.exists():
        st.warning(f"{state.output_path} doesn't exist and will be created!")

    segmentation_state_values(state)

    if st.button("Load model!") and state.model != None and state.use_gpu != None:
        state.net = UNet_Light_RDN(n_channels=1, n_classes=3)
        # Load in the trained model
        state.net.load_state_dict(
            torch.load(state.model, map_location=f"cuda:{state.use_gpu}")
        )
        state.net.cuda()
        state.net.eval()
        st.info("Model loaded!")
        save_state_values(state=state, user=current_user)

    if st.button("Segment!"):
        caching.clear_cache()
        failed_images = []
        st.warning(
            ":exclamation: The segmentation will stop if you navigate away from this page or interact with other "
            "buttons :frowning: :exclamation:"
        )
        net = state.net
        if net == None:
            st.error("You forgot to hit load model!")
            st.info(
                ":face_palm: Yes, this could be done automatically, but this step exists to make sure you are working "
                "with the model you want. Better this than wasting hours segmenting with the wrong thing."
            )
            st.stop()
        input_path = pathlib.Path(state.input_path)
        input_type = state.input_type

        output_path = pathlib.Path(state.output_path)
        out_type = state.out_type

        if not output_path.exists():
            pathlib.Path.mkdir(output_path)

        if state.input_type in slice_types:
            individual_seg_list.sort(key=natural_keys)
            st.write(
                f"Processing {len(individual_seg_list)} image files for segmentation in {input_path}..."
            )
            if bool(state.twoD_to_threeD) == False:
                three_class_segmentation(
                    input_image=individual_seg_list,
                    outDir=output_path,
                    outType=out_type,
                    network=state.net,
                )
            else:
                st.info(
                    f"Converting {input_type} stack to volume to be written as: {out_type}"
                )
                out_name = _get_file_name_from_list(
                    individual_seg_list, suffix="RDN_seg"
                )
                if input_type == "dcm":
                    image_vol, metadata = two_to_three(
                        image_stack=individual_seg_list, input_type=input_type
                    )
                else:
                    image_vol = two_to_three(
                        image_stack=individual_seg_list, input_type=input_type
                    )
                image_vol = rescale_8(image_vol)

                if state.rescale_prior == True:
                    if state.check_for_metal == False:
                        image_vol = rescale_before_seg(inputImage=image_vol)
                    else:
                        image_vol = rescale_before_seg(
                            inputImage=image_vol,
                            check_for_metal=True,
                            cut_off=int(state.cut_off),
                        )

                st.write("Segmenting...")
                seg_vol = three_class_seg_xyz(inputImage=image_vol, network=net)
                seg_vol.CopyInformation(image_vol)
                view_midplanes(image_volume=seg_vol, use_resolution=True)
                st.info("Writing image")
                write_image(
                    inputImage=seg_vol,
                    outName=out_name,
                    outDir=output_path,
                    fileFormat=out_type,
                )

        elif state.input_type in volume_types:
            out_name = _get_file_name_from_input(individual_seg, suffix="RDN_seg")
            try:
                image_vol = read_image(inputImage=individual_seg)
                image_vol = rescale_8(image_vol)

                if state.rescale_prior is True:
                    if state.check_for_metal is True:
                        cut_off = int(state.cut_off)
                        st.write(
                            f"Threshold for metal/bright inclusions set at {cut_off}"
                        )
                        image_vol = rescale_before_seg(
                            inputImage=image_vol, check_for_metal=True, cut_off=cut_off
                        )
                    else:
                        image_vol = rescale_before_seg(inputImage=image_vol)

                else:
                    pass
                st.write("Segmenting....")
                seg_vol = three_class_seg_xyz(inputImage=image_vol, network=net)

                # Just to be sure it gets the info
                seg_vol.CopyInformation(image_vol)
                view_midplanes(image_volume=seg_vol, use_resolution=True)
                write_image(
                    inputImage=seg_vol,
                    outName=out_name,
                    outDir=output_path,
                    fileFormat=out_type,
                )
            except RuntimeError:
                failed_images.append(str(individual_seg))
        else:
            st.error("Setting not understood!")
        st.write(failed_images)


def page_batch_segmentations(state):
    current_user = _get_user()
    segmentation_batch_values(state)
    try:
        if not state._state["data"]["model"]:
            st.error("Model not set!")
    except KeyError:
        st.error("Model not set!")

    if st.sidebar.button("Help!"):
        st.sidebar.info(
            "Note: If you place a '#' mark in the parameter file and reload it will ignore that line."
        )
        st.sidebar.info(
            "Note: Hitting the 'Stop' in the top right will kill the operation."
        )
        st.sidebar.markdown("**Steps:**")
        st.sidebar.markdown("1) Load the model")
        st.sidebar.markdown("2) Load parameter file to check it over")
        st.sidebar.markdown("3) Check the 'everything looks ok' box")
        st.sidebar.markdown("4) Segment!")

    st.write("---")
    state.rescale_prior = False
    state.check_for_metal = False
    if st.checkbox("Everything looks ok"):
        st.warning(
            "After hitting 'Segment', hitting other buttons or navigating away from this page will terminate "
            "the segmentation. :confounded:"
        )

        if st.checkbox("Rescale to 255 prior to segmentation?"):
            state.rescale_prior = True
            if st.checkbox("Check for metal/bright inclusions?", key="9990619"):
                state.check_for_metal = True
                state.cut_off = st.text_input("Threshold out values above:", 200)
        else:
            state.rescale_prior = False
            state.check_for_metal = False

        if st.button("Segment"):
            failed_images = []
            batch = state.parms
            net = state.net
            batch_len = int(len(state.parms))
            curr_batch = batch_len
            st.sidebar.text(f"Currently segmenting {batch_len} images:")

            with st.spinner("Running segmentations..."):
                for row in batch.itertuples():
                    st.sidebar.text(f"{curr_batch} of {batch_len} remaining.")
                    state.twoD_to_threeD = False
                    st.write(row.input_path)
                    input_path = pathlib.Path(row.input_path)
                    input_type = str(row.input_type)
                    output_path = pathlib.Path(row.output_path)
                    out_type = row.output_type
                    if not output_path.exists():
                        st.warning(f"{output_path} doesn't exist and will be created!")
                        pathlib.Path.mkdir(output_path)

                    _check_for_slice_to_vol(
                        state=state,
                        input_type=input_type,
                        out_type=out_type,
                        slice_types=slice_types,
                        volume_types=volume_types,
                    )

                    if state.twoD_to_threeD == True:
                        image_files = gather_image_files(
                            input_path=input_path, input_type=input_type
                        )
                        out_name = _get_file_name_from_list(
                            image_files, suffix="RDN_seg"
                        )
                        if input_type == "dcm":
                            image_vol, metadata = two_to_three(
                                image_stack=image_files, input_type=input_type
                            )
                        else:
                            image_vol = two_to_three(
                                image_stack=image_files, input_type=input_type
                            )
                    elif input_type in volume_types:
                        input_name = str(row.input_name)
                        out_name = _get_file_name_from_input(
                            input_name, suffix="RDN_seg"
                        )
                        full_input = input_path.joinpath(input_name)
                        try:
                            image_vol = read_image(inputImage=str(full_input))
                            image_vol = rescale_8(image_vol)
                            # This is ad-hoc because I keep fucking forgetting I did this for imagej.
                            # border = thresh_simple(inputImage=image_vol, background=254, foreground=255, outside=0)
                            # image_vol = subtract_images(inputImage1=image_vol, inputImage2=border)
                        except RuntimeError:
                            failed_images.append(str(full_input))
                    else:
                        st.error(f"Input type {input_type} not supported!")

                    if state.rescale_prior is True:
                        if state.check_for_metal is True:
                            cut_off = int(state.cut_off)
                            st.write(
                                f"Threshold for metal/bright inclusions set at {cut_off}"
                            )
                            image_vol = rescale_before_seg(
                                inputImage=image_vol,
                                check_for_metal=True,
                                cut_off=int(cut_off),
                            )
                        else:
                            image_vol = rescale_before_seg(inputImage=image_vol)
                    try:
                        image_vol = image_vol
                        execute_xyz_RDN_seg(
                            input_vol=image_vol,
                            network=net,
                            output_path=output_path,
                            out_name=out_name,
                            out_type=out_type,
                        )
                    except:
                        st.error(f"Failed to segment {out_name}")
                    curr_batch -= 1
            st.success("All segmentations are finished! :smile:")
            if failed_images:
                st.write("Failed segmentations:", failed_images)
                failed_images = pd.DataFrame(failed_images)
                failed_segs = str(pathlib.Path(state.parm_file).as_posix()).replace(
                    ".csv", "_failed_segmentations.csv"
                )
                failed_images.to_csv(failed_segs)
                st.info(f"Failed segmentation file written as {failed_segs}")

    else:
        st.info(
            "Make sure to load the model and check the parameter file before confirming that everything looks ok. "
        )
        st.write("---")
        if st.button("Load model!"):
            state.net = UNet_Light_RDN(n_channels=1, n_classes=3)
            # Load in the trained model
            state.net.load_state_dict(
                torch.load(state.model, map_location=f"cuda:{state.use_gpu}")
            )
            state.net.cuda()
            state.net.eval()
            st.info("Model loaded!")

        if st.button("Load parameter file"):
            state.parms = read_parameter_file(state.parm_file)
            save_state_values(state=state, user=current_user)
        if st.checkbox("View parameters file"):
            if state.parms is None:
                st.error("Please define the parameter file in settings!")
            else:
                if st.checkbox("Wrap style", value=True):
                    st.subheader("Batch parameters")
                    st.table(state.parms.style.highlight_null(null_color="red"))
                else:
                    st.subheader("Batch parameters")
                    st.dataframe(state.parms.style.highlight_null(null_color="red"))


#####
#
# Midslice viewer
#
#####


def page_midslice_viewer(state):
    current_user = _get_user()
    temp_state = _get_state()
    temp_state.compare_selected = "False"
    st.title("RDN segmentation viewer")
    midslice_radio_list = [
        "Compare single segmentations",
        "Compare from batch file",
        "View isolated file",
    ]

    midslice_settings_activity = st.experimental_get_query_params()
    # Query parameters are returned as a list to support multiselect.
    # Get the first item in the list if the query parameter exists.
    midslice_default = (
        int(midslice_settings_activity["activity"][0])
        if "activity" in midslice_settings_activity
        else 0
    )
    midslice_settings_activity = st.radio(
        "Select comparative style", midslice_radio_list, index=midslice_default
    )

    st.sidebar.markdown("## Viewing options ##")

    st.write("---")
    if midslice_settings_activity == "Compare single segmentations":
        if st.checkbox("Use saved input", value=False):
            st.write(state.input_path)
            if state.input_path != "None":
                temp_state.unseg_path = state.input_path
                temp_state.unseg_type = state.input_type
                temp_state.unseg_location = file_selector(
                    folder_path=temp_state.unseg_path,
                    extension=str(temp_state.unseg_type),
                )

                st.write(f"Unsegmented file: {temp_state.unseg_location}")
            else:
                st.error("No saved unsegmented input path found")
        else:
            temp_state.unseg_path = pathlib.Path(
                st.text_input("Unsegmented input path")
            )
            temp_state.unseg_type = st.selectbox(
                "Unsegmetned file type", supported_file_types
            )
            temp_state.unseg_location = file_selector(
                folder_path=temp_state.unseg_path, extension=str(temp_state.unseg_type)
            )
        if st.checkbox("Use saved output", value=False):
            if str(state.output_path) == "None" or str(state.output_path) == ".":
                st.error("No saved segmented output path found")
            else:
                temp_state.seg_path = state.output_path
                temp_state.seg_type = state.out_type
                temp_state.seg_location = file_selector(
                    folder_path=temp_state.seg_path,
                    extension=str(temp_state.seg_type),
                    unique_key="9990722",
                )

        else:
            temp_state.seg_path = pathlib.Path(st.text_input("Segmented input path"))
            temp_state.seg_type = st.selectbox(
                "Segmented file type", supported_file_types
            )
            temp_state.seg_location = file_selector(
                folder_path=temp_state.seg_path,
                extension=str(temp_state.seg_type),
                unique_key="999688",
            )

    if midslice_settings_activity == "Compare from batch file":
        if st.button("Load parameter file"):
            if state.parm_file == None:
                pass
            else:
                temp_state.parm_file = state.parm_file
                temp_state.batch_df = read_parameter_file(temp_state.parm_file)

        if st.checkbox("View parameter file"):
            st.write("Loaded", (temp_state.parm_file))
            if st.checkbox("Wrap style"):
                st.subheader("Batch parameters")
                st.table(temp_state.batch_df)
            else:
                st.subheader("Batch parameters")
                st.dataframe(temp_state.batch_df)

        st.write("---")
        st.write(temp_state.batch_df)
        if temp_state.batch_df is not None:
            batch_df = temp_state.batch_df
            input_options = list(batch_df["input_path"].unique())
            input_file_options = list(batch_df["input_type"].unique())
            output_options = list(batch_df["output_path"].unique())
            output_file_options = list(batch_df["output_type"].unique())

            if len(input_options) > 1:
                temp_state.unseg_path = st.selectbox(
                    "Unsegmented location", input_options
                )
            else:
                temp_state.unseg_path = input_options[0]
                # st.info(f"Unsegmented path set to {temp_state.unseg_path}")

            if len(input_file_options) > 1:
                temp_state.unseg_type = st.selectbox(
                    "Choose file input_type", input_file_options
                )
            else:
                temp_state.unseg_type = input_file_options[0]
                # st.info(f"Unsegmented file type set to {temp_state.unseg_type}")

            if len(output_options) > 1:
                temp_state.seg_path = st.selectbox("Segmented location", output_options)
            else:
                temp_state.seg_path = output_options[0]
                # st.info(f"Segmented path set to {temp_state.seg_path}")

            if len(output_file_options) > 1:
                temp_state.seg_type = st.selectbox(
                    "Choose file input_type", output_file_options
                )
            else:
                temp_state.seg_type = output_file_options[0]
                # st.info(f"Unsegmented file type set to {temp_state.seg_type}")

            temp_state.unseg_location = file_selector(
                folder_path=temp_state.unseg_path, extension=str(temp_state.unseg_type)
            )

            # The unique key is just to circumvent creating two widgets with the same id
            temp_state.seg_location = file_selector(
                folder_path=temp_state.seg_path,
                extension=temp_state.seg_type,
                unique_key="999901",
            )

    ##
    image_zoom = st.sidebar.text_input("Image scale:", 1.0)
    if temp_state.unseg_type == None:
        pass

    elif temp_state.unseg_type in slice_types:
        temp_state.unseg_vol_file = gather_image_files(
            input_path=temp_state.seg_location, input_type=temp_state.unseg_type
        )
        # TODO write this up to work with single slices
        temp_state.unseg_name = _get_file_name_from_list(temp_state.unseg_vol_file)
    else:
        temp_state.unseg_vol_file = temp_state.unseg_location

    if temp_state.unseg_type == None and temp_state.seg_type == None:
        st.write("Please choose your unsegmented and segmented files to compare")

    elif temp_state.unseg_type == None or temp_state.seg_type == None:
        st.write("Please choose your unsegmented and segmented files to compare")
    else:
        if str(temp_state.unseg_type) != "nan":
            st.write(temp_state.unseg_vol_file)
            temp_state.unseg_type = temp_state.unseg_type
            if str(temp_state.unseg_vol_file) != "None":
                st.info(
                    f"{pathlib.Path(temp_state.unseg_vol_file)} ready to load :smile:"
                )
    st.write("---")
    view_style = st.radio("Views:", ["Comparative", "Single view"])

    if st.button("Clear comparison"):
        caching.clear_cache()
        temp_state.unseg_vol_file = ""
        temp_state.stack_loaded = False

    if view_style == "Comparative":
        if st.checkbox("View comparison", key="9990827"):
            if temp_state.unseg_vol_file == "":
                st.error("Please select an input and output")
                st.stop()
            with st.spinner("Loading volumes..."):
                if temp_state.unseg_type in slice_types:
                    temp_state.stack_loaded = True
                    stack_input = pathlib.Path(temp_state.unseg_location)
                    stack_list = glob.glob(
                        str(stack_input.joinpath(f"*.{temp_state.unseg_type}"))
                    )
                    stack_list.sort(key=natural_keys)
                    st.write(
                        f"Processing {len(stack_list)} image files from {stack_input}..."
                    )

                    if temp_state.unseg_type == "dcm":
                        unseg_vol, metadata = two_to_three(
                            image_stack=stack_list, input_type=temp_state.unseg_type
                        )
                    else:
                        unseg_vol = two_to_three(
                            image_stack=stack_list, input_type=temp_state.unseg_type
                        )

                    unseg_vol = rescale_8(unseg_vol)
                    seg_vol = generate_single_vol_data(vol_file=temp_state.seg_location)

                elif temp_state.unseg_type in volume_types:
                    vols = generate_vol_data(
                        unseg_vol_file=str(temp_state.unseg_vol_file),
                        seg_vol_file=temp_state.seg_location,
                    )
                    unseg_vol, seg_vol = vols

                # unseg_vol = sitk.ReadImage(temp_state.unseg_vol_file)
                unseg_dims = unseg_vol.GetSize()

                # image_vol = sitk.ReadImage(str(seg_location))
                seg_dims = seg_vol.GetSize()
                slice_direction = "z"
                d_index = _direction_index(slice_direction)
                if unseg_dims[d_index] != seg_dims[d_index]:
                    st.error(
                        f":fearful: {slice_direction} dimension isn't the same for the two images! :fearful:"
                    )
                    caching.clear_cache()
                    raise st.ScriptRunner.StopException

            slice_direction = st.sidebar.selectbox("Slice direction", ["x", "y", "z"])
            d_index = _direction_index(slice_direction)

            compare_slice_num = st.sidebar.slider(
                label=f"{slice_direction} slice number",
                min_value=0,
                max_value=unseg_dims[d_index],
                value=int(unseg_dims[d_index] * 0.5),
            )

            unseg = get_midplane(
                image_volume=unseg_vol,
                slice_num=compare_slice_num,
                direction=str(slice_direction),
            )
            seg = get_midplane(
                image_volume=seg_vol,
                slice_num=compare_slice_num,
                direction=str(slice_direction),
            )

            compare_style = st.radio("Views:", ["Side by side", "Overlay", "XYZ"])
            if compare_style == "Side by side":
                view_midplane(
                    image_slice=[unseg, seg],
                    slice_res=unseg_vol.GetSpacing()[0],
                    use_resolution=True,
                    zoom_scale=image_zoom,
                )
                # view_midplane(image_slice, slice_res, use_resolution=False)
                # st.image([unseg, seg], width=int(unseg_dims[d_index] * float(image_zoom)), use_column_width=False)

            if compare_style == "Overlay":
                st.sidebar.markdown("## Overlay options ##")
                overlay_thresh = st.sidebar.text_input("Overlay thresh level", 200)
                overlay_opacity = st.sidebar.slider(
                    label=f"Overlay opacity", min_value=0.0, max_value=1.0, value=0.5
                )
                try:
                    color = st.sidebar.color_picker(
                        label="Overlay Color", value="#ff1493"
                    )
                except StreamlitAPIException:
                    color = st.sidebar.color_picker(
                        label="Overlay Color", value="#ff1493"
                    )
                rgb_color = ImageColor.getrgb(str(color))

                bottom_layer = cv2.cvtColor(
                    np.array(unseg).astype(np.uint8), cv2.COLOR_GRAY2BGR
                )
                top_layer = cv2.cvtColor(
                    np.array(seg).astype(np.uint8), cv2.COLOR_GRAY2BGR
                )

                overlay = color_overlay(
                    image=bottom_layer,
                    overlay_image=top_layer,
                    overlay_thresh=int(overlay_thresh),
                    color=list(rgb_color),
                    alpha=float(overlay_opacity),
                    darkmode=False,
                )
                view_midplane(
                    image_slice=overlay,
                    slice_res=unseg_vol.GetSpacing()[0],
                    use_resolution=True,
                    zoom_scale=image_zoom,
                )

            if compare_style == "XYZ":
                # Probably drop this into a funciton later.
                d_index_x = _direction_index("x")
                compare_slice_x = st.slider(
                    label=f"x slice number",
                    min_value=0,
                    max_value=unseg_dims[d_index_x],
                    value=int(unseg_dims[d_index_x] * 0.5),
                    key="99999000",
                )
                unseg_x = feed_slice(
                    inputImage=unseg_vol, slice=compare_slice_x, direction=str("x")
                )
                seg_x = feed_slice(
                    inputImage=seg_vol, slice=compare_slice_x, direction=str("x")
                )

                d_index_y = _direction_index("y")
                compare_slice_y = st.slider(
                    label=f"y slice number",
                    min_value=0,
                    max_value=unseg_dims[d_index_y],
                    value=int(unseg_dims[d_index_y] * 0.5),
                    key="99999001",
                )
                unseg_y = feed_slice(
                    inputImage=unseg_vol, slice=compare_slice_y, direction=str("y")
                )
                seg_y = feed_slice(
                    inputImage=seg_vol, slice=compare_slice_y, direction=str("y")
                )

                d_index_z = _direction_index("z")
                compare_slice_z = st.slider(
                    label=f"z slice number",
                    min_value=0,
                    max_value=unseg_dims[d_index_z],
                    value=int(unseg_dims[d_index_z] * 0.5),
                    key="99999002",
                )
                unseg_z = feed_slice(
                    inputImage=unseg_vol, slice=compare_slice_z, direction=str("z")
                )
                seg_z = feed_slice(
                    inputImage=seg_vol, slice=compare_slice_z, direction=str("z")
                )

                unseg_array_x = sitk.GetArrayFromImage(unseg_x).astype(np.uint8)
                unseg_array_y = sitk.GetArrayFromImage(unseg_y).astype(np.uint8)
                unseg_array_z = sitk.GetArrayFromImage(unseg_z).astype(np.uint8)

                unseg_x = Image.fromarray(unseg_array_x)
                unseg_y = Image.fromarray(unseg_array_y)
                unseg_z = Image.fromarray(unseg_array_z)

                seg_array_x = sitk.GetArrayFromImage(seg_x).astype(np.uint8)
                seg_array_y = sitk.GetArrayFromImage(seg_y).astype(np.uint8)
                seg_array_z = sitk.GetArrayFromImage(seg_z).astype(np.uint8)

                seg_x = Image.fromarray(seg_array_x)
                seg_y = Image.fromarray(seg_array_y)
                seg_z = Image.fromarray(seg_array_z)

                if st.checkbox("Overlay XYZ"):
                    st.sidebar.markdown("## Overlay options ##")
                    overlay_thresh = st.sidebar.text_input("Overlay thresh level", 200)
                    overlay_opacity = st.sidebar.slider(
                        label=f"Overlay opacity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                    )
                    try:
                        color = st.sidebar.color_picker(
                            label="Overlay Color", value="#ff1493"
                        )
                    except StreamlitAPIException:
                        color = st.sidebar.color_picker(
                            label="Overlay Color", value="#ff1493"
                        )
                    rgb_color = ImageColor.getrgb(str(color))

                    bottom_layer_x = cv2.cvtColor(unseg_array_x, cv2.COLOR_GRAY2BGR)
                    bottom_layer_y = cv2.cvtColor(unseg_array_y, cv2.COLOR_GRAY2BGR)
                    bottom_layer_z = cv2.cvtColor(unseg_array_z, cv2.COLOR_GRAY2BGR)
                    top_layer_x = cv2.cvtColor(seg_array_x, cv2.COLOR_GRAY2BGR)
                    top_layer_y = cv2.cvtColor(seg_array_y, cv2.COLOR_GRAY2BGR)
                    top_layer_z = cv2.cvtColor(seg_array_z, cv2.COLOR_GRAY2BGR)

                    overlay_x = color_overlay(
                        image=bottom_layer_x,
                        overlay_image=top_layer_x,
                        overlay_thresh=int(overlay_thresh),
                        color=list(rgb_color),
                        alpha=float(overlay_opacity),
                        darkmode=False,
                    )
                    overlay_y = color_overlay(
                        image=bottom_layer_y,
                        overlay_image=top_layer_y,
                        overlay_thresh=int(overlay_thresh),
                        color=list(rgb_color),
                        alpha=float(overlay_opacity),
                        darkmode=False,
                    )
                    overlay_z = color_overlay(
                        image=bottom_layer_z,
                        overlay_image=top_layer_z,
                        overlay_thresh=int(overlay_thresh),
                        color=list(rgb_color),
                        alpha=float(overlay_opacity),
                        darkmode=False,
                    )

                    st.image(
                        [overlay_x, overlay_y, overlay_z],
                        width=int(np.array(unseg_dims).max() * float(image_zoom)),
                        use_column_width=False,
                    )

                else:
                    st.image(
                        [unseg_x, unseg_y, unseg_z],
                        width=int(np.array(unseg_dims).max() * float(image_zoom)),
                        use_column_width=False,
                    )
                    st.image(
                        [seg_x, seg_y, seg_z],
                        width=int(np.array(unseg_dims).max() * float(image_zoom)),
                        use_column_width=False,
                    )
        st.sidebar.markdown(
            ":rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow:"
        )
        if st.sidebar.checkbox("Setup Sharon's magic mesh button "):
            unseg_file_name = (
                pathlib.Path(temp_state.unseg_location).parts[-1].split(".")[0]
            )
            suggested_name = _get_file_name_from_input(
                volume_file=unseg_file_name, suffix="Mesh"
            )
            mesh_name = st.sidebar.text_input("Output file name", str(suggested_name))
            mesh_out_type = st.sidebar.selectbox(
                "Mesh type", ["ply", "off", "vtk", "inp", "stl", "obj"]
            )
            mesh_out_dir = st.sidebar.text_input(
                "Output directory", str(pathlib.Path(temp_state.seg_path))
            )
            resample_amount = st.sidebar.selectbox("Resample by", list(range(1, 21)))
            try:
                res = seg_vol.GetSpacing()
                st.sidebar.markdown(
                    f"Element spacing will be {(res[0] * resample_amount):2.5f}"
                )
            except:
                pass
            try:
                thresh_amount = st.sidebar.text_input(
                    "Segmentation threshold", str(overlay_thresh), max_chars=3
                )
            except UnboundLocalError:
                thresh_amount = st.sidebar.text_input(
                    "Segmentation threshold", 128, max_chars=3
                )
            keep_largest = str(
                st.sidebar.selectbox("Keep largest component?", ["Yes", "No"])
            )

            prior_close = str(st.sidebar.selectbox("Close holes?", ["Yes", "No"]))
            if prior_close == "Yes":
                prior_close_size = st.sidebar.selectbox(
                    "Minimum closing kernel size", list(range(1, 26))
                )
            else:
                prior_close_size = 0

            if resample_amount >= 2:
                fill_holes = str(
                    st.sidebar.selectbox("Voting fill holes filter?", ["No", "Yes"])
                )
                if fill_holes == "Yes":
                    kernel_amount = st.sidebar.selectbox(
                        "Voting closing kernel size", list(range(2, 15))
                    )
                    majority_amount = st.sidebar.selectbox(
                        "Number of touching voxels required", list(range(0, 21))
                    )
                st.sidebar.info(
                    "Because of the time that voting fill takes it should only be used when the volume is "
                    "slow or resampling is sufficiently high (e.g. spacing of ~ 0.5 spacing)"
                )

            if st.sidebar.button("MESH!"):
                res = seg_vol.GetSpacing()
                if resample_amount == 1:
                    vtk_image = vtk_read_mhd(inputImage=str(temp_state.seg_location))
                else:
                    resampled = seg_vol
                    resample_spacing = float(resample_amount) * res[0]
                    resampled = thresh_simple(
                        inputImage=resampled,
                        background=int(thresh_amount),
                        foreground=255,
                        outside=0,
                    )
                    resampled = rescale_intensity(
                        inputImage=resampled,
                        old_min=0,
                        old_max=1,
                        new_min=0,
                        new_max=1,
                        threads="threads",
                    )

                    st.write("Resampling...")
                    resampled = resample_sitk_image(
                        inputImage=resampled,
                        spacing=resample_spacing,
                        interpolator="linear",
                        fill_value=0,
                    )
                    if int(prior_close_size) != 0:
                        st.write("Closing...")
                        resampled = closing_morph(
                            inputImage=resampled,
                            closing_kernel=int(prior_close_size),
                            kernel_type="Sphere",
                            threads="threads",
                        )
                    if fill_holes == "Yes":
                        st.write("Filling holes...")
                        resampled = binary_voting_fill_iterative(
                            inputImage=resampled,
                            iterations=3,
                            radius=int(kernel_amount),
                            background=0,
                            foreground=1,
                            majority=int(majority_amount),
                            threads="threads",
                        )

                    temp_mhd_dir = pathlib.Path(tempfile.gettempdir())
                    write_image(
                        resampled,
                        outName="temp_mhd",
                        outDir=temp_mhd_dir,
                        fileFormat="mhd",
                    )
                    # passing directly to vtk won't work because there is some bug in the threading module.
                    # vtk_image = simpleitk_to_vtk(inputImage=resampled, outVol=None)
                    vtk_image = vtk_read_mhd(
                        inputImage=str(temp_mhd_dir.joinpath("temp_mhd.mhd"))
                    )
                if keep_largest == "Yes":
                    vtk_mesh = vtk_MarchingCubes(
                        inputImage=vtk_image, threshold=1, extract_largest=True
                    )
                else:
                    vtk_mesh = vtk_MarchingCubes(
                        inputImage=vtk_image, threshold=1, extract_largest=False
                    )
                vtk_mesh = pv.wrap(vtk_mesh)
                mesh_out = pathlib.Path(mesh_out_dir).joinpath(
                    f"{mesh_name}.{mesh_out_type}"
                )
                st.write(f"Writing out {mesh_out}")
                if mesh_out_type in ["ply", "vtk", "stl"]:
                    vtk_mesh.save(f"{str(mesh_out)}", vtk_mesh)
                else:
                    pv.save_meshio(f"{str(mesh_out)}", vtk_mesh)

            st.markdown("Meshing done! :smile:")
        st.sidebar.markdown(
            ":rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow:"
        )

        if st.sidebar.checkbox("Histogram"):
            plotly_hist = get_plotly_hist(image_array=unseg)
            st.plotly_chart(plotly_hist, use_container_width=False)
        else:
            st.info("Remember to hit 'clear comparison' to load another volume.")

    st.write("---")

    if midslice_settings_activity == "View isolated file":
        if st.checkbox("View unsegmented"):
            unseg_input = st.selectbox("Unsegmented input", batch_df["input_path"])
            st.write(unseg_input)
            if st.checkbox("From stack"):
                stack_type = st.selectbox("", slice_types)
                if st.checkbox("Load unseg stack"):
                    stack_files = gather_image_files(
                        input_path=unseg_input, input_type=stack_type
                    )
                    unseg_name = _get_file_name_from_list(stack_files)
                    unseg_vol = generate_single_vol_data(vol_file=stack_files)
                    dims = unseg_vol.GetSize()
                    unseg_slice_num = st.slider("Unseg Position", 0, dims[2])
                    unseg_image = feed_slice(
                        inputImage=unseg_vol, slice=unseg_slice_num, direction="z"
                    )
                    unseg = Image.fromarray(sitk.GetArrayFromImage(unseg_image))
                    st.image(unseg, caption=f"{unseg_name}", use_column_width=False)

            else:
                unseg_image = file_selector(folder_path=unseg_input)
                if st.button("Load unseg"):
                    unseg_name = pathlib.Path(unseg_image).parts[-1]
                    unseg = Image.open(unseg_image)
                    st.image(unseg, caption=f"{unseg_name}", use_column_width=False)

        if st.checkbox("View segmented"):
            segmented_input = st.selectbox("Segmented input", batch_df["output_path"])
            seg_location = file_selector(folder_path=segmented_input, extension="mhd")

            if st.checkbox("Load segmented"):
                seg_name = pathlib.Path(seg_location).parts[-1]
                image_vol = sitk.ReadImage(str(seg_location))
                dims = image_vol.GetSize()
                seg_slice_num = st.slider("Seg Position", 0, dims[2])
                seg_image = feed_slice(
                    inputImage=image_vol, slice=seg_slice_num, direction="z"
                )
                seg = Image.fromarray(sitk.GetArrayFromImage(seg_image))
                st.image(seg, caption=f"{seg_name}", use_column_width=False)


def page_batch_meshing(state):
    st.title("Batch RDN Meshing")
    current_user = _get_user()
    mesh_state = _get_state()
    if st.sidebar.button("Reload parameter file", key="99991194"):
        mesh_state.parms = read_parameter_file(state.parm_file)

    if st.sidebar.button("Help!"):
        st.sidebar.info("Note: TODO.")

    st.write("---")
    if st.button("Load parameter file", key="9991105"):
        mesh_state.parms = read_parameter_file(state.parm_file)
    if st.checkbox("View parameters file"):
        if mesh_state.parms is None:
            st.error("Please define the parameter file in settings!")
        else:
            if st.checkbox("Wrap style", value=True):
                st.subheader("Batch parameters")
                st.table(mesh_state.parms.style.highlight_null(null_color="red"))
            else:
                st.subheader("Batch parameters")
                st.dataframe(mesh_state.parms.style.highlight_null(null_color="red"))
    st.write("---")
    st.markdown(
        ":rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow:"
    )

    if st.checkbox("Setup Sharon's magic mesh button "):
        mesh_state.mesh_name_append = st.text_input(
            "Append to mesh name", "_RDN_Mesh", key="9991113"
        )
        mesh_state.mesh_out_type = st.selectbox(
            "Mesh type", ["ply", "off", "vtk", "inp", "stl", "obj"]
        )

        if not mesh_state.mesh_out_dir:
            st.write("The output path from the parameter file will be used.")

        if st.checkbox("Override output dir?"):
            st.info(
                "This is useful if you suspect that you will need to post process a large number of meshes "
                "with the MARS meshlab app."
            )
            mesh_state.overide_output = True
            mesh_state.mesh_out_dir = st.text_input(
                "Output directory", str(pathlib.Path(mesh_state.seg_path))
            )
            try:
                if not pathlib.Path(str(mesh_state.mesh_out_dir)).exists():
                    st.warning(
                        f"{mesh_state.mesh_out_dir} doesn't exist and will be created!"
                    )
            except:
                st.write("Please paste ")

        else:
            mesh_state.overide_output = False

        st.write("---")
        # TODO put in the option to run the connected component filter prior to close
        if st.checkbox("Keep largest voxel structure prior to closing/resampling?"):
            mesh_state.isolate_largest_voxel = True
            st.warning(
                "A connected components filter finds the largest contiguous set of voxels. "
                "This can take a long time and uses lots of memory, but if elements are really close "
                "this will prevent them getting 'glued' together during closing/resampling steps. Note that a "
                "more aggressive close may be needed to compensate for the obliterated internal features."
            )
            if st.checkbox("Saved isolated volume", key="999901203"):
                st.info(
                    "Isoalted volume will be saved in the output folder with '_isolated' appended to the name."
                )
                mesh_state.save_isolated = True
            else:
                mesh_state.save_isolated = False
            if st.checkbox("Additional info", key="999901208"):
                st.info(
                    "The connected voxel threshold is the number of connected voxels required to be a 'valid' portion"
                    " Think of it as saying, I want you to ignore pieces that are smaller than what you set. "
                    "This can be much larger than you may expect. A 100 x 100 x 100 volume contains 1 million "
                    "voxels."
                )
            mesh_state.min_connect_voxels = int(
                st.text_input("Minimum connected voxel threshold", 10000, key="99901209")
            )
        else:
            mesh_state.isolate_largest_voxel = False
            mesh_state.min_connect_voxels = None

        if st.checkbox("Keep largest mesh component?"):
            mesh_state.keep_largest = True
            if st.checkbox("Additional info", key=999901215):
                st.info(
                    "This occurs after meshing and considers the largest connected faces in a mesh. It is fast, and "
                    "removes and small bits that are present in the volume. Even if you isolate the largest voxel "
                    "structure it may be necessary to use this if you want to move directly to remeshing/cleaning."
                )
        else:
            mesh_state.keep_largest = False
        st.write("---")

        # TODO make it keep going if the meshing fails, then provide a list at the end wof which ones failed
        if st.checkbox("Resample volume spacing prior to meshing?", key=9991139):
            if st.checkbox("Use resample column?", key=99901226):
                st.info(
                    "This is useful if you need distinct parameters for elements of a different size."
                )
                mesh_state.resample_column = True
                mesh_state.resample_amount = 999999
                if st.button("Create resample column", key=1208):
                    _save_altered_parm(
                        current_state_parm=mesh_state.parm_file,
                        new_column="resample_amount",
                        column_value=5,
                        dtype=float,
                    )
            else:
                mesh_state.resample_amount = st.selectbox(
                    "Resample by", list(range(2, 21)), key=99901208
                )
                mesh_state.resample_column = False
        else:
            mesh_state.resample_amount = 0

        st.write("---")

        if st.checkbox("Use threshold column?", key=99901205):
            st.info(
                "This is useful if you have complex segmentations that won't mesh properly with a single value."
            )
            mesh_state.threshold_column = True
            if st.button("Create threshold column", key=99901208):
                _save_altered_parm(
                    current_state_parm=mesh_state.parm_file,
                    new_column="mesh_threshold",
                    column_value=128,
                    dtype=int,
                )
        else:
            mesh_state.threshold_column = False
            mesh_state.thresh_amount = st.text_input(
                "Segmentation threshold", 128, max_chars=3, key=99901237
            )
        st.write("---")

        prior_close = str(st.selectbox("Close holes?", ["Yes", "No"]))
        if prior_close == "Yes":
            if st.checkbox("Use closing kernel column?", key=99901232):
                mesh_state.closing_column = True
                if st.button("Create closing kernel column", key=99901234):
                    _save_altered_parm(
                        current_state_parm=mesh_state.parm_file,
                        new_column="closing_kernel",
                        column_value=3,
                        dtype=int,
                    )
            else:
                mesh_state.prior_close_size = st.selectbox(
                    "Maximum closing kernel size", list(range(1, 26)), key=99901241
                )
                mesh_state.closing_column = False

        else:
            mesh_state.prior_close_size = 0

        if mesh_state.resample_amount >= 2:
            mesh_state.fill_holes = str(
                st.selectbox("Voting fill holes filter?", ["No", "Yes"], key=9991153)
            )
            if mesh_state.fill_holes == "Yes":
                mesh_state.voting_fill = True
                mesh_state.kernel_amount = st.selectbox(
                    "Voting closing kernel size", list(range(2, 21))
                )
                mesh_state.majority_amount = st.selectbox(
                    "Number of touching voxels required", list(range(0, 30))
                )
                st.info(
                    "Because of the time that voting fill takes it should only be used when the volume is "
                    "slow or resampling is sufficiently high (e.g. spacing of ~ 0.5 spacing)"
                )
            else:
                mesh_state.voting_fill = False

        st.write("---")
        if st.checkbox("Everything is set the way I want", key=9991141):
            batch = pd.read_csv(mesh_state.parm_file, comment="#")
            st.warning(
                "After hitting the 'Mesh!' button, hitting other buttons or navigating away from this page will terminate "
                "progress. :confounded:"
            )
            if st.button("Mesh!", key=9991117):
                batch_len = int(len(batch))
                st.sidebar.text(f"Currently meshing {batch_len} volumes:")
                with st.spinner(f"Meshing {batch_len} volumes..."):
                    progress_bar = st.progress(0)
                    current_total = 0
                    remaining_items = batch_len
                    for row in batch.itertuples():
                        st.sidebar.text(f"{remaining_items} of {batch_len} remaining.")
                        st.write(row.output_path)
                        input_name = str(row.input_name).rsplit(".", 1)[0]
                        input_path = pathlib.Path(row.output_path)
                        out_type = row.output_type

                        if mesh_state.resample_column == False:
                            resample_amount = mesh_state.resample_amount
                        else:
                            resample_amount = row.resample_amount

                        if mesh_state.overide_output:
                            mesh_out_dir = pathlib.Path(mesh_state.mesh_out_dir)
                            if not mesh_out_dir.exists():
                                pathlib.Path.mkdir(mesh_out_dir)
                        else:
                            mesh_out_dir = pathlib.Path(row.output_path)

                        if mesh_state.threshold_column:
                            thresh_amount = row.mesh_threshold
                        else:
                            thresh_amount = mesh_state.thresh_amount

                        if mesh_state.closing_column:
                            closing_kernel = row.closing_kernel
                        else:
                            closing_kernel = mesh_state.prior_close_size

                        st.info(f"Meshing {input_name}")
                        segmentation_name = input_path.joinpath(
                            f"{input_name}_RDN_seg.{out_type}"
                        )

                        if mesh_state.isolate_largest_voxel:
                            seg_vol = read_image(str(segmentation_name), verbose=True)
                            seg_vol = isolate_largest_bone(
                                seg_vol, min_size=int(mesh_state.min_connect_voxels)
                            )
                            if mesh_state.save_isolated:
                                isolated_name = f"{input_name}_isolated"
                            else:
                                isolated_name = "temp_isolated"
                                temp_isolated = str(
                                    input_path.joinpath(f"{isolated_name}.{out_type}")
                                )

                            seg_vol = rescale_8(seg_vol, verbose=False)
                            write_image(
                                inputImage=seg_vol,
                                outName=f"{isolated_name}",
                                outDir=str(input_path),
                                fileFormat=out_type,
                            )

                        if (
                            float(resample_amount) == 1.0
                            and mesh_state.isolate_largest_voxel
                        ):
                            isolated_seg = input_path.joinpath(
                                f"{isolated_name}.{out_type}"
                            )
                            vtk_image = vtk_read_mhd(inputImage=str(isolated_seg))
                        elif float(resample_amount) == 1.0:
                            vtk_image = vtk_read_mhd(inputImage=str(segmentation_name))
                        else:
                            if mesh_state.isolate_largest_voxel:
                                isolated_seg = input_path.joinpath(
                                    f"{isolated_name}.{out_type}"
                                )
                                seg_vol = read_image(inputImage=str(isolated_seg))
                            else:
                                seg_vol = read_image(str(segmentation_name))
                            res = seg_vol.GetSpacing()
                            st.info(
                                f"Element spacing will be {(res[0] * resample_amount):2.5f}"
                            )
                            resample_spacing = float(resample_amount) * res[0]
                            resampled = thresh_simple(
                                inputImage=seg_vol,
                                background=int(thresh_amount),
                                foreground=255,
                                outside=0,
                            )
                            seg_vol = 0  # Clean up memory

                            resampled = rescale_intensity(
                                inputImage=resampled,
                                old_min=0,
                                old_max=1,
                                new_min=0,
                                new_max=1,
                                threads="threads",
                            )

                            st.write("Resampling...")
                            resampled = resample_sitk_image(
                                inputImage=resampled,
                                spacing=resample_spacing,
                                interpolator="linear",
                                fill_value=0,
                            )

                            if int(closing_kernel) != 0:
                                st.write("Closing...")
                                resampled = closing_morph(
                                    inputImage=resampled,
                                    closing_kernel=int(closing_kernel),
                                    kernel_type="Sphere",
                                    threads="threads",
                                )

                            if mesh_state.voting_fill:
                                st.write("Filling holes...")
                                resampled = binary_voting_fill_iterative(
                                    inputImage=resampled,
                                    iterations=3,
                                    radius=int(mesh_state.kernel_amount),
                                    background=0,
                                    foreground=1,
                                    majority=int(mesh_state.majority_amount),
                                    threads="threads",
                                )

                            temp_mhd_dir = pathlib.Path(tempfile.gettempdir())
                            write_image(
                                resampled,
                                outName="temp_mhd",
                                outDir=temp_mhd_dir,
                                fileFormat="mhd",
                            )
                            # passing directly to vtk won't work because there is some bug in the threading module.
                            # vtk_image = simpleitk_to_vtk(inputImage=resampled, outVol=None)
                            vtk_image = vtk_read_mhd(
                                inputImage=str(temp_mhd_dir.joinpath("temp_mhd.mhd"))
                            )

                            # Clean up files so they don't pile up
                            file_clean_up = pathlib.Path(
                                str(temp_mhd_dir.joinpath("temp_mhd.mhd"))
                            )
                            try:
                                file_clean_up.unlink(missing_ok=True)
                            except:
                                pass
                            try:
                                temp_isolated.unlink(missing_ok=True)
                            except:
                                pass

                        vtk_mesh = vtk_MarchingCubes(
                            inputImage=vtk_image,
                            threshold=1,
                            extract_largest=bool(mesh_state.keep_largest),
                        )
                        vtk_mesh = pv.wrap(vtk_mesh)
                        mesh_out = pathlib.Path(mesh_out_dir).joinpath(
                            f"{input_name}{mesh_state.mesh_name_append}.{mesh_state.mesh_out_type}"
                        )
                        st.write(f"Writing out {mesh_out}")
                        if mesh_state.mesh_out_type in ["ply", "vtk", "stl"]:
                            vtk_mesh.save(f"{str(mesh_out)}", vtk_mesh)
                        else:
                            pv.save_meshio(f"{str(mesh_out)}", vtk_mesh)
                        current_total += 1
                        remaining_items -= 1
                        iteration = np.floor(100 * (current_total / batch_len))
                        progress_bar.progress(int(iteration))
                    st.markdown("Meshing done! :smile:")
    else:
        st.markdown(
            ":rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow: :sparkles: :rainbow:"
        )

    mesh_batch_values(mesh_state)


##
#
#  Things that are always displayed
#
###


def display_state_values(state):
    if ".pth" in str(state.model):
        st.info(f"Model path: {state.model}")
    else:
        st.error(f"Model path not set!")

    if str(state.parm_file) == "None":
        st.warning("Parameter file not set")
    else:
        st.info(f"Parameter file: {state.parm_file}")

    if str(state.input_path) == ".":
        pass
    else:
        st.write("Input path:", state.input_path)

    st.info(f"Input file type {state.input_type}")

    if str(state.output_path) == ".":
        pass
    else:
        st.write("Output path:", state.output_path)

    st.info(f"Output file type {state.out_type}")


def segmentation_state_values(state):
    st.title("RDN 3-class segmentation:")
    if str(state.cuda_mem) == "None":
        st.error("GPU not initialized!")
    else:
        st.info(
            f"Segmenting with {torch.cuda.get_device_name(state.use_gpu)}, {state.cuda_mem} of memory."
        )

    if ".pth" in str(state.model):
        st.info(f"Model path: {state.model}")
    else:
        st.error(f"Model path not set!")
    st.info(
        f"Segmentations will be written to {str(state.output_path)} in {state.out_type} file format"
    )

    _check_for_slice_to_vol(
        state=state,
        input_type=str(state.input_type),
        out_type=str(state.out_type),
        slice_types=slice_types,
        volume_types=volume_types,
    )
    if bool(state.twoD_to_three) == True:
        st.info(
            f"{state.input_type} is assumed to be slices and will be converted to 3d output type: {state.out_type}"
        )


def segmentation_batch_values(state):
    st.title("Batch RDN 3-class segmentation:")
    if str(state.cuda_mem) == "None":
        st.error("GPU not initialized!")
    else:
        st.info(
            f"Segmenting with {torch.cuda.get_device_name(state.use_gpu)}, {state.cuda_mem} of memory."
        )
    if ".pth" in str(state.model):
        st.info(f"Model path: {state.model}")
    else:
        st.error(f"Model path not set!")
    st.info(f"Parameter file: {state.parm_file}")


def mesh_batch_values(mesh_state):
    # st.write(mesh_state._state["data"])
    st.title("Batch RDN Meshing:")
    if not mesh_state.parm_file:
        st.error("No parameter file set for meshing!")
    else:
        st.info(
            f"Meshing volumes contained in the {mesh_state.parm_file} parameter file."
        )

    if mesh_state.overide_output:
        st.info(f"Mesh files will be written to {mesh_state.mesh_out_dir}")
    st.info(
        f"Mesh file names will have *{mesh_state.mesh_name_append}* appended to each."
    )
    st.info(f"Mesh files will be written in {mesh_state.mesh_out_type} file format.")

    if not mesh_state.resample_amount:
        st.info("No resampling set")
    elif int(mesh_state.resample_amount) == 999999:
        st.info(
            f"Segmented volume will be resampled according to the parameter files 'resample_amount' column."
        )
    else:
        st.info(f"Segmented volume will be resampled by {mesh_state.resample_amount}")

    if mesh_state.threshold_column:
        st.info(
            "Segmented volumes will be thresholded according to the 'mesh_threshold' column."
        )
    else:
        st.info(f"Threshold set at {mesh_state.thresh_amount}.")

    if mesh_state.closing_column:
        st.info(
            "Segmented volumes will have a closing filter applied according to the 'closing_kernel' column."
        )
    elif mesh_state.prior_close_size != 0:
        st.info(
            f"Segmentation holes will be closed using a spherical kernel size of {mesh_state.prior_close_size}"
        )
    else:
        st.info("No closing will be performed prior to meshing.")

    if mesh_state.fill_holes == "Yes":
        st.info(
            f"Voting closing will of {int(mesh_state.kernel_amount)} will be applied if "
            f"{int(mesh_state.majority_amount)} voxels are touching."
        )

    if mesh_state.keep_largest:
        st.info(f"A mesh of the largest connected elements will be extracted.")
    else:
        st.info(f"Not extracting the largest connected mesh element.")


####
#
# Code for this streamlit app
#
####
def _custom_logo(image_path, image_text):
    st.sidebar.markdown(
        """<style>
        figcaption {
                   font-weight:100 !important;
                   font-size:25.5px !important;
                   text-align: left !important;
                   color: black !important;
                   z-index: 1;             
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"""<figure>
                                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" alt="my img" width="300"/>
                                <figcaption>{image_text}</figcaption>
                            </figure>""",
                        unsafe_allow_html=True)

@st.cache_data
def _load_MARS_logo():
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent
    moon_logo = script_dir.joinpath('streamlit_apps').joinpath('Moon_logo_small.png')
    _custom_logo(image_path=moon_logo, image_text="RDN 3-Class Segmentation ")
    


def _get_tab_logo():
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    tab_logo = Image.open(str(script_dir.joinpath("moon_square.png")))
    return tab_logo


def _get_user():
    """
    Internal function to get the username for saving the settings.
    :return: Returns the operating system username using os.environ
    """
    # We can't always count on this being launchde from a C: on windows
    # So we get the current working directory, then if there are back slashes we grab the root drive letter.
    current = pathlib.Path.cwd()
    if "\\" in str(current):
        windows_drive = str(current.parts[0])
    pc = socket.gethostname()
    user = [
        os.environ["USERNAME"]
        if str(windows_drive) in os.getcwd()
        else os.environ["USER"]
    ]
    user = f"{user[0]}_{pc}"
    return user


def _get_file_name_from_list(image_files, suffix=""):
    outName = pathlib.Path(image_files[0]).parts[-1]
    outName = outName.split(".")[0]
    outName = outName.rsplit("_", 1)[0]
    if suffix != "":
        outName = f"{outName}_{suffix}"
    return outName


def _get_file_name_from_input(volume_file, suffix="RDN_seg"):
    if "\\" or "/" in volume_file:
        outName = pathlib.Path(volume_file).parts[-1]
    outName = outName.split(".")[0]
    if suffix != "":
        outName = f"{outName}_{suffix}"
    return outName


def _check_for_slice_to_vol(
    state, input_type, out_type, slice_types, volume_types, verbose=False
):
    if input_type in slice_types and out_type in volume_types:
        if verbose == True:
            st.info(
                f"Slice input type: {input_type} and volume output type: {out_type} selected."
            )
            st.info(
                f" This is assumed to be an image stack and the slices will be converted to a volume output."
            )
        state.twoD_to_threeD = True
        current_user = _get_user()
        save_state_values(state=state, user=current_user)
    else:
        state.twoD_to_threeD = False


def _direction_index(slice_direction="z"):
    slice_direction = str(slice_direction).lower()
    if slice_direction == "z":
        return int(2)
    elif slice_direction == "y":
        return int(1)
    else:
        return int(0)


def _alter_parameter(parameter_file, new_column, column_value):
    current_df = pd.read_csv(
        parameter_file,
        index_col=0,
        dtype={
            "input_path": "str",
            "output_path": "str",
            "input_type": "str",
            "output_type": "str",
        },
    )
    if new_column not in current_df.columns:
        st.write(f"Adding '{new_column}' column to ", parameter_file)
        st.write(f"'{new_column}' starting value set at ", column_value)
        current_df.index.set_names("")
        current_df[f"{new_column}"] = column_value
        return current_df
    else:
        st.error(f"{new_column} already in the parameter file.")
        return False


def _save_altered_parm(current_state_parm, new_column, column_value, dtype):
    current_df = _alter_parameter(
        parameter_file=current_state_parm,
        new_column=f"{new_column}",
        column_value=column_value,
    )
    if type(current_df) is pd.DataFrame:
        current_df[f"{new_column}"] = current_df[f"{new_column}"].astype(dtype)
        current_df.to_csv(f"{current_state_parm}")


@st.cache(
    allow_output_mutation=True,
    hash_funcs={builtins.tuple: lambda _: None},
    suppress_st_warning=True,
)
def generate_vol_data(unseg_vol_file, seg_vol_file):
    unseg_vol = sitk.ReadImage(unseg_vol_file)
    unseg_vol = rescale_8(inputImage=unseg_vol, verbose=False)
    seg_vol = sitk.ReadImage(str(seg_vol_file))
    return unseg_vol, seg_vol


@st.cache(
    allow_output_mutation=True,
    hash_funcs={builtins.tuple: lambda _: None},
    suppress_st_warning=True,
)
def generate_single_vol_data(vol_file):
    vol_file = read_image(vol_file, verbose=False)
    vol_file = rescale_8(inputImage=vol_file, verbose=False)
    return vol_file


def is_state_value_empty(state_value, verbose=False):
    """
    Small function to check if a state value is empty, None, nan, or other non-sense
    :param state_value:
    :param verbose:
    :return:
    """
    if verbose:
        st.write(state_value)
    if str(state_value) == "None" or str(state_value) == ".":
        if verbose:
            st.error("State value empty")
        return True
    else:
        if verbose:
            st.write("State value not empty")
        return False


def get_state_path(current_state, current_user, key, message=""):
    """
    WIP. Should be a small function to deal with a lot of redundancy
    :param state_value:
    :param key:
    :param message:
    :return:
    """
    state = current_state
    if st.checkbox("Previous input path", value=True, key=9990363):
        if is_state_value_empty(state.input_path, verbose=False):
            st.warning("No saved path found, please enter a new location")
            state.output_path = pathlib.Path(st.text_input("Output location", ""))
            save_state_values(state=state, user=current_user)
        else:
            state.input_path = pathlib.Path(state.input_path)
            st.write(state.input_path)
            save_state_values(state=state, user=current_user)
    else:
        state.input_path = pathlib.Path(st.text_input("Input location", ""))
        save_state_values(state=state, user=current_user)


def save_state_values(state, user):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(
        f"{user}_RDN_saved_state.json"
    )
   
    session_dict = {
        "model_path": [str(state.model_path)],
        "model": [str(state.model)],
        "input_path": [str(state.input_path)],
        "input_type": [state.input_type],
        "output_path": [str(state.output_path)],
        "out_type": [str(state.out_type)],
        "parm_file": [str(state.parm_file)],
    }
    session_dict = pd.DataFrame.from_dict(session_dict)
    session_dict.to_json(saved_dir)
   
    st.write(f"Saved settings for {user}!")


def load_state_values(state, user, verbose=True):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(
        f"{user}_RDN_saved_state.json"
    )
    restored_session = pd.read_json(str(saved_dir))
    if verbose:
        st.write("Restored previous settings")
    state.model_path = str(restored_session["model_path"][0])
    state.model = restored_session["model"][0]
    state.input_path = restored_session["input_path"][0]
    state.input_type = restored_session["input_type"][0]
    state.output_path = pathlib.Path(restored_session["output_path"][0])
    state.out_type = restored_session["out_type"][0]
    state.parm_file = restored_session["parm_file"][0]
    return state


def file_selector(folder_path=".", extension="", selectbox_text="", unique_key=""):
    import random
    if folder_path == "." or "":
        folder_path = pathlib.Path.cwd()
    filenames = os.listdir(folder_path)
    filenames.sort(reverse=True)
    if extension != "":
        filenames = [num for num in filenames if extension in num]
    if unique_key == "":
        selected_filename = st.selectbox(f"{selectbox_text}", filenames )
    else:
        selected_filename = st.selectbox(f"{selectbox_text}", filenames, key=unique_key)
    try:
        return os.path.join(folder_path, selected_filename)
    except:
        pass


def gather_image_files(input_path, input_type):
    image_files = glob.glob(str(pathlib.Path(input_path).joinpath(f"*.{input_type}")))
    image_files.sort(key=natural_keys)
    st.write(
        f"Found {len(image_files)} image files for segmentation in {input_path}..."
    )
    return image_files


def read_parameter_file(parm_file):
    parms = pd.read_csv(parm_file, index_col=0, comment="#")
    return parms


def parm_from_directory(
    input_dir, input_file_type="tif", output_dir="", output_file_type="mhd"
):
    # Directory to be scanned
    input_dir = pathlib.Path(input_dir)
    if input_file_type in slice_types:
        dir_obj = os.scandir(input_dir)
        dir_list = [
            str(input_dir.joinpath(entry.name)) for entry in dir_obj if entry.is_dir
        ]
        dir_list.sort(key=natural_keys)
        parm_file = pd.DataFrame(dir_list)
        parm_file.columns = ["input_path"]
    else:
        dir_list = glob.glob(str(input_dir.joinpath(f"*.{input_file_type}").as_posix()))
        # dir_list.sort(key=natural_keys)
        dir_list.sort(key=os.path.getmtime)
        parm_file = pd.DataFrame(dir_list)
        parm_file[0] = parm_file[0].str.replace("\\", "/")
        parm_file = parm_file[0].str.rsplit("/", n=1, expand=True)
        parm_file.columns = ["input_path", "input_name"]
    parm_file["input_type"] = str(input_file_type)
    if output_dir == "":
        parm_file["output_path"] = (
            parm_file["input_path"].astype(str).map("{}_seg".format)
        )
    else:
        parm_file["output_path"] = output_dir
    parm_file["output_type"] = str(output_file_type)
    return parm_file


def parm_from_par(par_file, out_type="mhd", input_type="mhd"):
    parm_file = pd.read_csv(par_file, sep=";")
    parm_file = parm_file[["$path", "$oldname"]]
    parm_file["$oldname"] = parm_file["$oldname"].str.replace("#", "")
    parm_file.columns = ["input_path", "input_name"]
    parm_file["input_path"] = parm_file["input_path"].map(
        lambda x: str(pathlib.Path(x).as_posix())
    )
    parm_file["input_path"] = parm_file["input_path"].str.cat(
        "/" + parm_file["input_name"]
    )
    parm_file["input_type"] = str(input_type)
    parm_file["output_path"] = (
        parm_file["input_path"].astype(str).map("{}/01_Seg".format)
    )
    parm_file["input_path"] = (
        parm_file["input_path"].astype(str).map("{}/00_Original".format)
    )
    parm_file["input_name"] = parm_file["input_name"].str.cat(
        "." + parm_file["input_type"]
    )
    parm_file["output_type"] = str(out_type)
    return parm_file


def two_to_three(image_stack, input_type):
    image_stack.sort(key=natural_keys)
    st.write("Converting image stack...")
    if input_type == "dcm":
        image_vol, metadata = read_dicom(image_stack)
        resolution = image_vol.GetSpacing()
        if resolution[2] == 1.0:
            st.write(
                f"Setting z resolution to {resolution[1]} to match x and y resolution...."
            )
            image_vol.SetSpacing((resolution[0], resolution[1], resolution[1]))
        return image_vol, metadata
    elif input_type in ["tif", "png", "jpg", "bmp"]:
        image_vol = read_stack(image_stack)
        resolution = image_vol.GetSpacing()
        image_vol.SetSpacing((resolution[0], resolution[1], resolution[1]))
        return image_vol
    else:
        st.markdown(f"Input type {input_type} not supported :frowning:")


####
#
#   Model utils
#
#####


def initiate_cuda(state):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        st.write("No cuda device found!")
    elif num_gpus > 1:
        st.write("Multiple gpu's found...")
        state.use_gpu = gpu_selector(num_gpus=num_gpus)
    else:
        state.use_gpu = 0
    torch.cuda.set_device(state.use_gpu)
    state.cuda_mem = (
        str(torch.cuda.get_device_properties(state.use_gpu))
        .split(",")[-2]
        .split("=")[-1]
    )


def gpu_selector(num_gpus):
    gpu_list = list(range(num_gpus))
    selected_gpu = st.selectbox("Select the gpu", gpu_list)
    return selected_gpu


def model_initiation(model_path, cuda_index):
    net = UNet_Light_RDN(n_channels=1, n_classes=3)
    # Load in the trained model
    net.load_state_dict(torch.load(model_path, map_location=f"cuda:{int(cuda_index)}"))
    net.cuda()
    net.eval()
    return net


###################################
#
#       Streamlit visuals
#
###################################


def get_midplane(image_volume, slice_num, direction="x"):
    image_slice = feed_slice(
        inputImage=image_volume, slice=slice_num, direction=direction
    )
    image_array = sitk.GetArrayFromImage(image_slice).astype(np.uint8)
    image_slice = Image.fromarray(image_array)
    return image_slice


def view_midplane(image_slice, slice_res, use_resolution=False, zoom_scale=1):
    if type(image_slice) == list:
        seg_dims = int(np.array([image.size[0] for image in image_slice]).max())
    else:
        seg_dims = image_slice.shape[0]

    if use_resolution:
        image_zoom = int((seg_dims * slice_res) * float(zoom_scale)) * 5
    else:
        image_zoom = None
    st.image(image_slice, clamp=True, width=image_zoom, use_column_width=False)


def view_midplanes(image_volume, use_resolution=False):
    image_volume.CopyInformation(image_volume)
    seg_x, seg_y, seg_z = get_xyz_midplanes(image_volume=image_volume)
    seg_dims = image_volume.GetSize()
    seg_res = image_volume.GetSpacing()[0]
    if use_resolution:
        image_zoom = int(np.array([int(seg_res * x) for x in seg_dims]).max())
    else:
        image_zoom = None
    st.image(
        [seg_x, seg_y, seg_z],
        caption=["x", "y", "z"],
        clamp=True,
        width=image_zoom,
        use_column_width=False,
    )


def get_plotly_hist(image_array):
    hist_data = image_array.flatten()
    plotly_hist = px.histogram(
        hist_data,
        labels={"x": "Intensity values", "y": "count"},
        nbins=255,
        color_discrete_sequence=["#ff1493"],
        width=400,
        height=200,
    )

    xticks = np.array(list(range(0, 255, 25)))
    plotly_hist.update_xaxes(tickvals=list(xticks), range=[0, 255])
    plotly_hist.update_yaxes(showticklabels=False, showgrid=False)  # , type="log")
    plotly_hist.update_layout(
        showlegend=False,
        paper_bgcolor="Black",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title_text=None,
        yaxis_title_text=None,
    )
    return plotly_hist


def get_xyz_midplanes(image_volume):
    """
    Function to get the midplanes of an image volume for streamlit.
    :param image_volume:
    :return:
    """
    image_mids = [int(dim * 0.5) for dim in image_volume.GetSize()]
    compare_slice_num_x, compare_slice_num_y, compare_slice_num_z = image_mids
    image_x = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_x, direction="x"
    )
    image_y = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_y, direction="y"
    )
    image_z = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_z, direction="z"
    )
    array_x = sitk.GetArrayFromImage(image_x).astype(np.uint8)
    array_y = sitk.GetArrayFromImage(image_y).astype(np.uint8)
    array_z = sitk.GetArrayFromImage(image_z).astype(np.uint8)
    image_x = Image.fromarray(array_x)
    image_y = Image.fromarray(array_y)
    image_z = Image.fromarray(array_z)
    return image_x, image_y, image_z


def get_midplane_histogram(image_volume, log=False):
    image_volume = generate_single_vol_data(vol_file=image_volume)
    # image_volume = read_image(inputImage=image_volume, verbose=False)
    # image_volume = rescale_8(image_volume, verbose=False)
    image_mids = [int(dim * 0.5) for dim in image_volume.GetSize()]
    compare_slice_num_x, compare_slice_num_y, compare_slice_num_z = image_mids
    image_x = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_x, direction="x"
    )
    image_y = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_y, direction="y"
    )
    image_z = feed_slice(
        inputImage=image_volume, slice=compare_slice_num_z, direction="z"
    )
    array_x = sitk.GetArrayFromImage(image_x).flatten()
    array_y = sitk.GetArrayFromImage(image_y).flatten()
    array_z = sitk.GetArrayFromImage(image_z).flatten()

    plotly_hist = go.Figure()
    plotly_hist.add_trace(
        go.Histogram(x=array_x, name="x-plane", marker_color="#ff1493", opacity=0.75)
    )
    plotly_hist.add_trace(
        go.Histogram(x=array_y, name="y-plane", marker_color="#7CFC00", opacity=0.75)
    )
    plotly_hist.add_trace(
        go.Histogram(x=array_z, name="z-plane", marker_color="#5CE9FF", opacity=0.75)
    )
    plotly_hist.update_layout(barmode="overlay")

    # Reduce opacity to see both histograms
    plotly_hist.update_traces(opacity=0.75)
    xticks = np.array(list(range(0, 255, 25)))
    plotly_hist.update_xaxes(tickvals=list(xticks), range=[0, 255])
    if log:
        plotly_hist.update_yaxes(showticklabels=False, showgrid=False, type="log")
    else:
        plotly_hist.update_yaxes(showticklabels=False, showgrid=False)  # , type="log")
    plotly_hist.update_layout(
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title_text=None,
        yaxis_title_text=None,
    )

    plotly_hist.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Courier", size=16, color="black"),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
        )
    )
    st.plotly_chart(plotly_hist, use_container_width=False)


def render_svg(svg_file):
    with open(svg_file, "r") as f:
        lines = f.readlines()
        svg = "".join(lines)
        """Renders the given svg string."""
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        return html


def custom_html(body):
    st.sidebar.markdown(body, unsafe_allow_html=True)


def card_begin_str(header):
    return (
        "<style>div.card{background-color:lightblue;border-radius: 5px;box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>"
        '<div class="card">'
        '<div class="container">'
        f"<h3><b>{header}</b></h3>"
    )


def card_end_str():
    return "</div></div>"


def card(header, body):
    lines = [card_begin_str(header), f"<p>{body}</p>", card_end_str()]
    custom_html("".join(lines))


def br(n):
    custom_html(n * "<br>")


####
#
#
# Modified code from MARS CLI
#
#
###


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


def _file_size(dim1, dim2, dim3, bits):
    """
    Get the file size of an image volume from the x, y, and z dimensions.
    :param dim1: The x dimension of an image file.
    :param dim2: The y dimension of an image file.
    :param dim3: The z dimension of an image file.
    :param bits: The type of bytes being used (e.g. unsigned 8 bit, float 32, etc.)
    :return: Returns the size in bytes.
    """
    if bits == 8:
        bit = 1
    elif bits == 16:
        bit = 2
    elif bits == 32:
        bit = 4
    else:
        bit = 8
    file_size = dim1 * dim2 * dim3 * bit
    file_s = _convert_size(sizeBytes=file_size)
    st.text(f"file size: {file_s[0]}")
    size = int(np.ceil(file_s[1]))
    return size


def _print_info(inputImage):
    """
    Function to return the basic information of an image volume.
    :param inputImage: A SimpleITK formated image volume.
    """
    image_type = inputImage.GetPixelIDTypeAsString()
    size = inputImage.GetSize()
    xdim, ydim, zdim = size[0], size[1], size[2]
    xres, yres, zres = inputImage.GetSpacing()
    if image_type == "8-bit unsigned integer":
        bits = 8
    elif image_type == "16-bit unsigned integer" or "16-bit signed integer":
        bits = 16
    elif image_type == "32-bit unsigned integer" or "32-bit signed integer":
        bits = 32
    else:
        bits = 64
    _file_size(xdim, ydim, zdim, bits)
    st.text(f"{image_type}")
    st.text(f"Dimensions, x:{xdim:5}, y:{ydim:5}, z:{zdim:5}")
    st.text(f"Resolution, x:{xres:5}, y:{yres:5}, z:{zres:5}")


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


def _end_timer(start_timer, message=""):
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
    if message == "":
        st.text(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        st.text(f"{message} took: {float(elapsed):10.4f} seconds")


def _get_threads(threads):
    if threads == "threads":
        threads = int(multiprocessing.cpu_count()) - 1
    else:
        threads = int(threads)
    return threads


def _get_kernel_type(kernel_type):
    kernel_type = kernel_type.lower()
    if kernel_type == "sphere":
        kernel_type = 1
    elif kernel_type == "cube":
        kernel_type = 2
    elif kernel_type == "cross":
        kernel_type = 3
    else:
        kernel_type = 4
    return kernel_type


def _connected_component(inputImage, threads="threads"):
    start = timer()
    cc = sitk.ScalarConnectedComponentImageFilter()
    cc.SetNumberOfThreads(_get_threads(threads))
    cc_image = cc.Execute(inputImage)
    _end_timer(start, message="Running connected component filter")
    return cc_image


def _scalar_connected_component(inputImage, min_size=100000, threads="threads"):
    start = timer()
    cc = _connected_component(inputImage, threads=threads)
    cc = sitk.RelabelComponent(
        image1=cc, minimumObjectSize=min_size, sortByObjectSize=True
    )
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, inputImage)
    _end_timer(start, message="Running scalar connected component")
    return cc, stats


def _check_for_metal(inputImage, cut_off=220):
    sitk_array = sitk.GetArrayFromImage(inputImage)
    sitk_array.max()
    intensity_vals, counts = np.unique(sitk_array, return_counts=True)
    if 255 in intensity_vals:
        how_many = counts[-1] / np.sum(counts[:-1])
    elif 240 in intensity_vals:
        max_vals = np.where(intensity_vals > 240)[0][0]
        how_many = np.sum(counts[int(max_vals) :]) / np.sum(counts[: int(max_vals)])
    else:
        how_many = 1
    if float(how_many) < 0.001:
        st.text("Metal/bright inclusions make up less than 0.1% of image...")
        max_vals = np.where(intensity_vals > 220)[0][0]
        how_many = np.sum(counts[int(max_vals) :]) / np.sum(counts[: int(max_vals)])
        if float(how_many) < 0.001:
            st.text(
                f"Metal/bright inclusions above {cut_off} make up {how_many:0.10f}% of image..."
            )
            return True
        else:
            return False
    else:
        return False


def read_stack(inputStack):
    """
    Reads in a series of images and then places them into a SimpleITK volume format.
    :param inputStack: A stack of images (e.g. tif, png, etc).
    :return: Returns a SimpleITK formatted image object.
    """
    start = timer()
    st.write("Reading in files...")

    # Sort the image stack like a human would.
    inputStack.sort(key=natural_keys)

    # Use just the straight SimpleITK reader beause the MARS version is pretty verbose.
    inputStack = sitk.ReadImage(inputStack)
    _end_timer(start, message="Reading in the stack")
    _print_info(inputStack)
    return inputStack


def read_dicom(inputStack):
    """
    Specialized DICOM reader that preserves the metadata tags in the dicom files.
    :param inputStack: Dicom stack.
    :return: Returns a SimpleITK image and a list containing the metadata.
    note that tag 0020|000e is a unique identifier that may be modified in the returned metadata. Otherwise the data
    and time are used.
    """

    start = timer()
    st.write(f"Reading in {len(inputStack)} DICOM images...")

    inputStack.sort(key=natural_keys)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(inputStack)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    sitk_image = series_reader.Execute()

    _print_info(sitk_image)

    # Grab the metadata, Name, ID, DOB, etc.
    direction = sitk_image.GetDirection()
    tags_to_copy = [
        "0010|0010",
        "0010|0020",
        "0010|0030",
        "0020|000D",
        "0020|0010",
        "0008|0020",
        "0008|0030",
        "0008|0050",
        "0008|0060",
        "0028|0030",
    ]
    process_tag = ["0008|103e"]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    series_tag_values = [
        (k, series_reader.GetMetaData(0, k))
        for k in tags_to_copy
        if series_reader.HasMetaDataKey(0, k)
    ]

    modified_tags = [
        ("0008|0031", modification_time),
        ("0008|0021", modification_date),
        ("0008|0008", "DERIVED\\SECONDARY"),
        ("0020|000e", "" + modification_date + ".1" + modification_time),
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
    ]

    series_tag_values = series_tag_values + modified_tags

    # Inset the new processing data
    if series_reader.HasMetaDataKey(0, process_tag[0]) == True:
        series_tag_values = series_tag_values + [
            (
                "0008|103e",
                series_reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK",
            )
        ]
    else:
        series_tag_values = series_tag_values + [("0008|103e", "Processed-SimpleITK")]

    # To prevent the stacking of the same processing information
    if series_tag_values[-1] == (
        "0008|103e",
        "Processed-SimpleITK  Processed-SimpleITK",
    ):
        series_tag_values[-1] = ("0008|103e", "Processed-SimpleITK")
    _end_timer(start_timer=start, message="Reading DICOM stack")
    return sitk_image, series_tag_values


def read_image(inputImage, verbose=True):
    """
    Reads in various image file formats (mha, mhd, nia, nii, vtk, etc.) and places them into a SimpleITK volume format.
    :param inputImage: Either a volume (mhd, nii, vtk, etc.).
    :return: Returns a SimpleITK formatted image object.
    """
    if verbose:
        st.write(f"Reading in {inputImage}.")
    start = timer()
    inputImage = sitk.ReadImage(str(inputImage))
    if verbose:
        _end_timer(start, message="Reading in the image")
        _print_info(inputImage)
        st.write("\n")
    return inputImage


def vtk_read_mhd(inputImage):
    """

    :param inputImage:
    :return:
    """
    start = timer()
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(str(inputImage))
    reader.Update()
    vtk_image = reader.GetOutput()
    _end_timer(start, message="Reading MHD")
    return vtk_image


def write_image(inputImage, outName, outDir="", fileFormat="mhd", verbose=True):
    """
    Writes out a SimpleITK image in any supported file format (mha, mhd, nii, dcm, tif, vtk, etc.).
    :param inputImage: SimpleITK formated image volume
    :param outName: The file name
    :param outDir: The directory where the file should be written to. If not path is provided the current directory will
    be used.
    :param fileFormat: The desired file format. If no file format is provided, mhd will be used.
    :return: Returns an image file written to the hard disk.
    """
    start = timer()
    outName = str(outName)
    outDir = _get_outDir(outDir)
    fileFormat = str(fileFormat)

    fileFormat = fileFormat.replace(".", "")
    outputImage = pathlib.Path(outDir).joinpath(f"{outName}.{fileFormat}")
    if verbose:
        _print_info(inputImage)
    st.write(f"Writing {outName} to {outDir} as {fileFormat}.")
    sitk.WriteImage(inputImage, str(outputImage))
    _end_timer(start, message="Writing the image")


def write_dicom(inputImage, metadata, outName, outDir=""):
    """
    Function to write out dicoms using SimpleITK.
    :param inputImage: SimpleITK image to be wrriten in DICOM format.
    :param metadata: Tagged metadata for DICOM format. Should be a dictionary in the form of (tag: value).
    :param outName: Output name for the dcm images.
    :param outDir: Output directory for the dcm images. If an empty string is provided "", the current directory will
    be used.
    :return: A DICOM stack written to the hard disk.
    """

    # Modified from: https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    start = timer()
    series_tag_values = metadata
    if outDir == "":
        outDir = pathlib.Path.cwd()
    else:
        outDir = pathlib.Path(outDir)

    # Make is so the file name generator deal with these parts of the name
    if outName[-4] == ".dcm":
        outName = outName[:-4]

    if outName[-1] == "_":
        outName = outName[:-1]

    outName = pathlib.Path(outDir).joinpath(outName)
    slice_num = inputImage.GetDepth()

    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    digits_offset = int(len(str(slice_num)))

    for i in range(slice_num):
        image_slice = inputImage[:, :, i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, inputImage.TransformIndexToPhysicalPoint((0, 0, i)))),
        )
        #   Instance Number
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        writer.SetFileName(f"{str(outName)}_{i:0{int(digits_offset)}}.dcm")
        writer.Execute(image_slice)
    _end_timer(start, message="Writing DICOM slices")


def feed_slice(inputImage, slice, direction="Z"):
    """
    Function to write out a single slice from a SimpleITK volume.
    :param inputImage: SimpleITK formatted image.
    :param outName: The resulting image file name along with the file format.
    :param slice: The slice number along the Z dimension.
    :return: An image written into memoryy.
    """
    direction = str(direction).lower()

    if direction == "z":
        image_slice = inputImage[:, :, slice]
    elif direction == "y":
        image_slice = inputImage[:, slice, :]
    else:
        image_slice = inputImage[slice, :, :]
    return image_slice


def rescale_8(inputImage, verbose=True):
    """
    Takes in a SimpleITK image and rescales it to 8 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 8-bit SimpleITK formatted volume with gray values scaled between 0-255.
    """

    # Check to see if it is already unisgned 8 bit.
    imageType = inputImage.GetPixelID()
    if imageType == 1:
        if verbose:
            st.write("Image is already unsigned 8...")
        scaled_8 = inputImage

    # If it isn't, go ahead and rescale.
    else:
        st.write("Rescaling to unsigned 8...")
        start = timer()
        scaled_8 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)
        if verbose:
            _print_info(scaled_8)
            _end_timer(start, message="Rescaling to unsigned 8")
    return scaled_8


def rescale_16(inputImage):
    """
    Takes in a SimpleITK image and rescales it to 16 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 16-bit SimpleITK formatted volume with gray values scaled between 0-65,535.
    """
    imageType = inputImage.GetPixelID()
    if imageType == 3:
        st.write("Image is already unsigned 16...")
        scaled_16 = inputImage
    else:
        scaled_16 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt16)
    return scaled_16


def rescale_32(inputImage):
    """
    Takes in a SimpleITK image and rescales it to 32 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 16-bit SimpleITK formatted volume with 2^64 distinct gray values.
    """
    imageType = inputImage.GetPixelID()
    if imageType == 8:
        st.write("Image is already float 32...")
        scaled_32 = inputImage
    else:
        # Read in the other image and recast to float 32
        scaled_32 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkFloat32)
    return scaled_32


def rescale_intensity(
    inputImage, old_min, old_max, new_min, new_max, threads="threads", verbose=False
):
    """
    Rescales the intensity values of the inputImage. Reads the values between the old_min and old_max then rescales
    anything above that to the the new_min and new_max.

    :param inputImage:
    :param old_min:
    :param old_max:
    :param new_min:
    :param new_max:
    :return:

    Example:
    new_image = rescale_intensity(new_image, 0, 255, 0, 1)

    """
    start = timer()
    threads = int(_get_threads(threads))

    # Get the min and max of an image.
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.SetNumberOfThreads(int(threads))

    minmax.Execute(inputImage)
    if verbose:
        print(
            f"Old image min: {minmax.GetMinimum()} \nOld image max: {minmax.GetMaximum()}\n"
        )

    if minmax.GetMaximum() < old_max:
        print(
            "Image maximum is less than the old maximum, scaling won't do anything..."
        )
        rescaled = inputImage
    else:
        rescale = sitk.IntensityWindowingImageFilter()
        rescale.SetWindowMinimum(old_min)
        rescale.SetWindowMaximum(old_max)
        rescale.SetOutputMinimum(new_min)
        rescale.SetOutputMaximum(new_max)
        rescaled = rescale.Execute(inputImage)
    minmax.Execute(rescaled)
    if verbose:
        st.info(
            f"New image min: {minmax.GetMinimum()} \nNew image max: {minmax.GetMaximum()}\n"
        )

    _end_timer(start, message="Rescaling intensity values")
    return rescaled


def rescale_before_seg(inputImage, check_for_metal=False, cut_off=220):
    """
    Load in an 8-bit 3d image file volume with lower intensity values and rescales them to 255  max prior to segmentation.
    :param inputImage: SimpleITK volume.
    :return: Returns a rescaled volume.
    """

    if check_for_metal is True:
        checked = _check_for_metal(inputImage, cut_off=int(cut_off))
    else:
        checked = False
    if bool(checked) is True:
        st.write("Thresholding")
        inputImage = thresh_simple(
            inputImage=inputImage, background=0, foreground=int(cut_off), outside=0
        )

    MinMax = sitk.MinimumMaximumImageFilter()
    MinMax.Execute(inputImage)
    rescaleFilt = sitk.RescaleIntensityImageFilter()
    rescaleFilt.SetOutputMinimum(0)
    rescaleFilt.SetOutputMaximum(255)
    imageMax = MinMax.GetMaximum()
    if imageMax < 220.0:
        st.write(f"Max intensity value is {imageMax}, rescaling to 255...")
        # Rescale to prevent overflow
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
        rescaled = rescaleFilt.Execute(inputImage)
        rescaled = sitk.Cast(rescaled, sitk.sitkUInt8)
        return rescaled
    else:
        st.write(f"Max intensity value is {imageMax}, rescaling isn't necessary...")
        rescaled = inputImage
        return rescaled


def vtk_MarchingCubes(inputImage, threshold=1, extract_largest=True):
    """
    http://www.vtk.org/Wiki/VTK/Examples/Cxx/Modelling/ExtractLargestIsosurface
    """
    start = timer()
    st.write("Running marching cubes...")
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(inputImage)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()

    # To keep only the largest voxel cluster
    if extract_largest:
        confilter = vtk.vtkPolyDataConnectivityFilter()
        confilter.SetInputData(mc.GetOutput())
        confilter.SetExtractionModeToLargestRegion()
        confilter.Update()
        mesh = confilter.GetOutput()
    else:
        mesh = mc.GetOutput()
    st.write(
        f"Mesh has {mesh.GetNumberOfPieces()} components with {mesh.GetNumberOfCells()} cells and {mesh.GetNumberOfPoints()}\n"
    )
    _end_timer(start, message="Running marching cubes took")
    return mesh


def resample_sitk_image(inputImage, spacing=None, interpolator=None, fill_value=0):
    """
    Resamples an ITK image to a new grid. If no spacing is given, the resampling is done isotropically to the smallest
    value in the current spacing. This is usually the in-plane resolution. If not given, the interpolation is derived
    from the input data type. Binary input (e.g., masks) are resampled with a specified interpolator.
    Modified from (https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py)
    interpolation is chosen.
    :param inputImage: Either a SimpleITK image or a path to a SimpleITK readable file.
    :param spacing: A tuple of integers (0.03,0.03,03.03)
    :param interpolator : A string with the type of interpolation that should be used during the resampling.
    Options are nearest, linear, gaussian, label_gaussian, bspline, hamming_sinc, cosine_windowed_sinc,
    welch_windowed_sinc, 'lanczos_windowed_sinc'.
    :param fill_value: Integer to set any unfilled voxels to (e.g. 0 for a black background)
    :return A resampled SimpleITK image.
    -------
    SimpleITK image.
    """
    inputImage = inputImage
    spacing = float(spacing)
    interpolator = str(interpolator)
    fill_value = int(fill_value)

    # Create a dictionary so the human readable text is replaced with the sitk call.
    SITK_INTERPOLATOR_DICT = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "b-spline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
        "hamming_sinc": sitk.sitkHammingWindowedSinc,
        "label_gaussian": sitk.sitkLabelGaussian,
        "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
        "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
        "blackman_windowed_sinc": sitk.sitkBlackmanWindowedSinc,
    }
    start = timer()

    # If the inputImage isn't read into memory, it should be read in.
    if type(inputImage) == str:
        inputImage = read_image(inputImage)

    # Get the image information
    num_dim = inputImage.GetDimension()
    orig_size = np.array(inputImage.GetSize(), dtype=np.int)
    orig_origin = inputImage.GetOrigin()
    orig_pixelid = inputImage.GetPixelIDValue()
    orig_direction = inputImage.GetDirection()
    orig_spacing = np.array(inputImage.GetSpacing())

    # Set up the interpolator type.
    if interpolator == None:
        pixelid = inputImage.GetPixelIDValue()
        if pixelid not in (1, 3, 8):
            raise NotImplementedError(
                'Set "interpolator" manually, \ncan only infer for 8 and 16 unsigned, or 32-bit float!'
            )
        if pixelid == 1:  #  8-bit unsigned int
            st.write("No interpolator set, using linear...\n")
            interpolator = "linear"
        if pixelid in (3, 8):  #  16-bit unsigned and 32 bit float
            st.write("No interpolator set, using b-spline...\n")
            interpolator = "b-spline"

    # Set the interpolator from the dictionary
    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    # Get the spacing to use from the minimum value if it isn't set.
    if spacing == None:
        min_spacing = orig_spacing.min()
        st.write(f"No spacing set, using the minimum spacing of {min_spacing}...")
        new_spacing = [min_spacing] * num_dim

    if len([spacing]) == 1:
        new_spacing = (float(spacing), float(spacing), float(spacing))
    else:
        st.write(f"Apply spacing of {spacing}...")
        new_spacing = [float(s) for s in spacing]

    # New Size is the original * the spacing/new spacing
    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]  #  SimpleITK expects lists, not ndarrays

    # Use the simpleitk filter
    resample_filter = sitk.ResampleImageFilter()

    # Hacky fix for the version differences
    if sitk.SITK_ITK_VERSION_MAJOR < 5:
        st.write(
            "You have an older version of SimpleITK installed in your anaconda environment."
        )
        st.write(
            "This function has been made to be backwards compatible, but if you notice others don't work please"
            "email me at nbs49@psu.edu or contact me through my github https://github.com/NBStephens/"
        )

        resampled_inputImage = resample_filter.Execute(
            inputImage,
            new_size,
            sitk.Transform(),
            sitk_interpolator,
            orig_origin,
            new_spacing,
            orig_direction,
            fill_value,
            orig_pixelid,
        )
    else:
        resample_filter.SetSize(new_size)
        resample_filter.SetTransform(sitk.Transform())
        resample_filter.SetInterpolator(sitk_interpolator)
        resample_filter.SetOutputOrigin(orig_origin)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetOutputDirection(orig_direction)
        resample_filter.SetDefaultPixelValue(fill_value)
        resample_filter.SetOutputPixelType(orig_pixelid)
        resampled_inputImage = resample_filter.Execute(inputImage)

    _print_info(resampled_inputImage)
    _end_timer(start, message="Resampling")
    return resampled_inputImage


def simpleitk_to_vtk(inputImage, outVol=None):
    """
    https://github.com/dave3d/dicom2stl/blob/master/sitk2vtk.py

    :param inputImage:
    :return:
    """
    # Dictionary for mapping datatypes between the two libraries
    pixelmap = {
        sitk.sitkUInt8: vtk.VTK_UNSIGNED_CHAR,
        sitk.sitkInt8: vtk.VTK_CHAR,
        sitk.sitkUInt16: vtk.VTK_UNSIGNED_SHORT,
        sitk.sitkInt16: vtk.VTK_SHORT,
        sitk.sitkUInt32: vtk.VTK_UNSIGNED_INT,
        sitk.sitkInt32: vtk.VTK_INT,
        sitk.sitkUInt64: vtk.VTK_UNSIGNED_LONG,
        sitk.sitkInt64: vtk.VTK_LONG,
        sitk.sitkFloat32: vtk.VTK_FLOAT,
        sitk.sitkFloat64: vtk.VTK_DOUBLE,
        sitk.sitkVectorUInt8: vtk.VTK_UNSIGNED_CHAR,
        sitk.sitkVectorInt8: vtk.VTK_CHAR,
        sitk.sitkVectorUInt16: vtk.VTK_UNSIGNED_SHORT,
        sitk.sitkVectorInt16: vtk.VTK_SHORT,
        sitk.sitkVectorUInt32: vtk.VTK_UNSIGNED_INT,
        sitk.sitkVectorInt32: vtk.VTK_INT,
        sitk.sitkVectorUInt64: vtk.VTK_UNSIGNED_LONG,
        sitk.sitkVectorInt64: vtk.VTK_LONG,
        sitk.sitkVectorFloat32: vtk.VTK_FLOAT,
        sitk.sitkVectorFloat64: vtk.VTK_DOUBLE,
        sitk.sitkLabelUInt8: vtk.VTK_UNSIGNED_CHAR,
        sitk.sitkLabelUInt16: vtk.VTK_UNSIGNED_SHORT,
        sitk.sitkLabelUInt32: vtk.VTK_UNSIGNED_INT,
        sitk.sitkLabelUInt64: vtk.VTK_UNSIGNED_LONG,
    }
    img = inputImage
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    sitktype = img.GetPixelID()
    vtktype = pixelmap[sitktype]
    ncomp = img.GetNumberOfComponentsPerPixel()

    # there doesn't seem to be a way to specify the image orientation in VTK

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    i2_string = i2.tobytes()

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()

    dataImporter.CopyImportVoidPointer(i2_string, len(i2_string))

    dataImporter.SetDataScalarType(vtktype)

    dataImporter.SetNumberOfScalarComponents(ncomp)

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    # Set the new VTK image's parameters
    #
    dataImporter.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    dataImporter.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)

    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()

    # outVol and this DeepCopy are a work-around to avoid a crash on Windows
    if outVol is not None:
        outVol.DeepCopy(vtk_image)

    st.write(vtk_image)
    st.write(f" data type is {vtktype}")
    st.write(f"{ncomp} components")
    st.write(f"Size:   {size}")
    st.write(f"Origin:  {origin}")
    st.write(f"Spacing: {spacing}")
    st.write(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return vtk_image


def combine_images(inputImage1, inputImage2):
    """
    Function to combine two SimpleITK images using the Add filter.
    :param inputImage1: SimpleITK image.
    :param inputImage2: SimpleITK image.
    :return: Returns a single SimpleITK image.
    """
    combined = sitk.Add(inputImage1, inputImage2)
    return combined


def closing_morph(
    inputImage,
    closing_kernel=3,
    foreground_value=1,
    kernel_type="Sphere",
    threads="threads",
):
    """
    Function for the morphological closing of a binary image.

    :param inputImage: A SimpleITK image.
    :param closing_kernel: The size in voxels for the closing kernel.
    :param kernel: The kernel type used in the closing operation (i.e. sphere, cube, cross)
    :param threads: The number of cores/threads/processors to use. If not set all but one will be used.
    :return: Returns a SimpleITK formatted image with closed holes.
    """
    start = timer()
    threads = int(_get_threads(threads))
    kernel_type = int(_get_kernel_type(kernel_type))

    closing_morph = sitk.BinaryMorphologicalClosingImageFilter()
    closing_morph.SetKernelRadius(closing_kernel)
    closing_morph.SetNumberOfThreads(threads)
    closing_morph.SetKernelType(kernel_type)
    closing_morph.SetForegroundValue(foreground_value)
    closed_image = closing_morph.Execute(inputImage)
    _end_timer(start, message="Applying the closing morphological filter")
    return closed_image


def binary_voting_fill_iterative(
    inputImage,
    iterations=3,
    radius=3,
    background=0,
    foreground=1,
    majority=2,
    threads="threads",
):
    """
    Function to apply a binary voting operation to a segmented image. This will repeat until there are no changes or the
    maximum number of times, set by te user with the "iterations" argument.

    :param inputImage: A SimpleITK image.
    :param iterations: An integer specifying the maximum attempts that will be made to fill all the hole.
    :param radius: The radius in voxels of the voting fill kernel.
    :param majority: The number of voxels that must be touching the kernel to be closed.
    :return: Returns a SimpleITK image with closed holes.

    """
    # Start timer for filter.
    start = timer()

    # Get the amount of threads/cores/processors to use.
    threads = int(_get_threads(threads))

    # Define the paramterers for the filter
    inputImage = inputImage
    radius = int(radius)
    iterations = int(iterations)
    background = int(background)
    foreground = int(foreground)
    majority = int(majority)

    holefill = sitk.VotingBinaryIterativeHoleFillingImageFilter()

    holefill.SetMaximumNumberOfIterations(iterations)
    holefill.SetRadius(radius)
    holefill.SetNumberOfThreads(threads)
    holefill.SetBackgroundValue(background)
    holefill.SetForegroundValue(foreground)
    holefill.SetMajorityThreshold(majority)
    filled_image = holefill.Execute(inputImage)

    # End the timer.
    _end_timer(start, message="Applying the binary voting filter")
    return filled_image


def thresh_simple(inputImage, background=0, foreground=1, outside=0, threads="threads"):
    start = timer()
    thresh = sitk.ThresholdImageFilter()
    thresh.SetNumberOfThreads(_get_threads(threads))
    thresh.SetLower(background)
    thresh.SetUpper(foreground)
    thresh.SetOutsideValue(outside)
    threshold = thresh.Execute(inputImage)
    _end_timer(start, message="Simple threshold")
    return threshold


def isolate_largest_bone(inputImage, min_size=100000, threads="threads"):
    start = timer()
    st.info(f"Running connected components... ")
    cc, stats = _scalar_connected_component(
        inputImage, min_size=min_size, threads=threads
    )
    st.write(
        f"Found {stats.GetNumberOfLabels()} connected components, including the background..."
    )
    bone = sitk.Threshold(cc, int(2), int(2))
    st.write(f"Element bounds at {stats.GetBoundingBox(2)}")
    bone = rescale_intensity(
        inputImage=bone, old_min=0, old_max=1, new_min=0, new_max=1, threads="threads"
    )
    _end_timer(start, message=f"Isolating largest bone")
    return bone


def color_overlay(
    image,
    overlay_image,
    overlay_thresh=254,
    color=[100, 8, 58],
    alpha=0.5,
    darkmode=False,
):
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
    ret, mask = cv2.threshold(
        src=overlay_image,
        thresh=int(overlay_thresh),
        maxval=255,
        type=cv2.THRESH_BINARY,
    )
    mask[np.where((mask == [255, 255, 255]).all(axis=2))] = color
    img_out = cv2.addWeighted(src1=img_out, alpha=1, src2=mask, beta=alpha, gamma=0)
    return img_out


######
#
# Pytorch segmentation code
#
#####


def three_class_segmentation(input_image, outDir, outType, network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a SimpleITK readable format (e.g. "tif", "png", "jpg", "bmp", "mhd", "nii", etc.)
    :param input_image: A list of images to be segmented.
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :param network: The pytorch network to be used for the segmentation.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    start = timer()
    save_folder = outDir

    net = network

    # The file types that can be output along with the corresponding dictionary
    if pathlib.Path(outDir).exists() != True:
        pathlib.Path.mkdir(save_folder)

    # Get a list of files from the input folder using a list comprehension approach, then sort them numerically.
    image_names = input_image
    image_names.sort(key=natural_keys)

    st.write(f"Processing {len(image_names)} images...")

    progress_bar = st.progress(0)
    # Loop through the images in the folder and use the image name for the output name
    for i in range(len(image_names)):
        image_name = image_names[i]
        if "\\" or "/" in image_name:
            out_name = str(pathlib.Path(image_name).parts[-1])
        else:
            out_name = image_name
        if "." in out_name:
            out_name = out_name.rsplit(".", 1)[0]

        # Read the image in with pillow and set it as a numpy array for pytorch
        image = sitk.ReadImage(str(image_name))
        # Check if the image is a vector and extract the first component, if so.
        if image.GetPixelID() == 13:
            image = sitk.VectorIndexSelectionCast(image, 0)

        # Rescale the image to 8 bit if it isn't already
        if image.GetPixelID() != 1:
            image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)

        image = _setup_sitk_image(image, direction="z")
        st.write(image)
        # Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        # Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()
        st.write("line 2930")
        # Pass the predictions to be saved using pillow
        _save_predictors(
            pred=pred, save_folder=outDir, image_name=out_name, file_type=outType
        )
        iteration = np.floor((100 * ((i + 1) / len(image_names))))
        progress_bar.progress(int(iteration))

    st.write("\n\nSegmentations are done!\n\n")
    _end_timer(start_timer=start, message="Segmentations")


def _save_predictors(pred, save_folder, image_name, file_type):
    """
    Internal function to convert predictions to an image and save in an output folder.
    """
    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    # File type dictionary for pillow
    type_dict = {"tif": "TIFF", "png": "PNG", "jpg": "JPEG", "bmp": "BMP"}
    f_type = type_dict[str(file_type)]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_img = pred_img.astype(np.uint8)
    pred_img = Image.fromarray(pred_img, "L")
    if "." in image_name:
        image_name = image_name.replace(".", "")
    pred_name = f"{image_name}.{str(file_type)}"
    pred_img.save(os.path.join(save_folder, str(pred_name)), str(f_type))


def execute_xyz_RDN_seg(input_vol, network, output_path, out_name, out_type):
    """
    Helper function for input and output to the three_class_seg_xyz function. Checks the input type is correct, and
    copies over the metadata from the input prior to writing out the segmented volume.
    :param input_vol: SimpleITK image volume
    :param network: Pytorch segmentation network.
    :param output_path: The path where the image volume will be written out to.
    :param out_name:  The name of the written image volume.
    :param out_type: The SimpleITK writatble output file type (e.g. mhd, nii, etc.)
    :return: Writes a segmented image volume to the hard disk.
    """
    # Check to see is the input image type is unsigned 8 bit. If it isn't it will be rescaled as such.
    if input_vol.GetPixelID() != 1:
        input_vol = rescale_8(input_vol)

    # Streamlit style information written to the GUI console.
    st.info(f"Segmenting {out_name}...")

    # Segmentation occurs, then the input metadata is written to the segmented volume prior to being written out.
    seg_vol = three_class_seg_xyz(inputImage=input_vol, network=network)
    seg_vol.CopyInformation(input_vol)
    write_image(
        inputImage=seg_vol, outName=out_name, outDir=output_path, fileFormat=out_type
    )


def three_class_segmentation_volume(inputImage, direction="z", network=""):
    """
    Function to segment a SimpleITK volume using a pytorch model
    :param inputImage: The SimpleITK image volume
    :param direction: The image plane that you want the segmentation to be performed along.
    :param network: The Pytorch nueral network.
    :return: Returns a segmented image volume with grey values representing air, non-bone, and bone.
    """
    net = network
    start = timer()

    # The direction of the plan that will be segmented along.
    direction = str(direction).lower()

    if direction == "z":
        seg_count = inputImage.GetSize()[2]
    elif direction == "y":
        seg_count = inputImage.GetSize()[1]
    else:
        seg_count = inputImage.GetSize()[0]
    st.write(f"Processing {seg_count} {direction} slices...")

    # Create an empty volume to stuff the results into. A numpy approach was tested but proved to be slower
    vol_image = sitk.Image(inputImage.GetSize(), sitk.sitkUInt8)

    # Streamlit progress bar for the gui
    progress_bar = st.progress(0)

    # Loop through the images in the folder and use the image name for the output name
    for i in range(seg_count):
        image = feed_slice(inputImage, slice=i, direction=str(direction))

        # Read the image in with pillow and set it as a numpy array for pytorch
        image = _setup_sitk_image(image_slice=image, direction=direction)

        # Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        # Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()

        # Pass the predictions to be saved using pillow.
        pred = _return_predictors(pred=pred, direction=direction)
        slice_vol = sitk.GetImageFromArray(pred)
        # slice_vol = sitk.JoinSeries(slice)
        if direction == "z":
            vol_image = sitk.Paste(
                vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, 0, i]
            )
        elif direction == "y":
            vol_image = sitk.Paste(
                vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, i, 0]
            )
        else:
            vol_image = sitk.Paste(
                vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[i, 0, 0]
            )

        iteration = np.floor((100 * ((i + 1) / seg_count)))
        progress_bar.progress(int(iteration))

    _end_timer(start_timer=start, message=f"{direction}-plane segmentations")
    return vol_image


def three_class_seg_xyz(inputImage, network=""):
    """
    Function to do RDN segmentation along the Z, Y, and X planes.
    :param inputImage: SimpleITK volume.
    :param network: Pytorch convolutional network.
    :return: Returns a segmented SimpleITK volume with grey values representing air, non-bone, and bone.
    """
    # Streamlit progress bar so everyone knows that it is running.
    progress_bar = st.progress(0)

    # The progress bar takes inegers and there are six steps in this process.
    steps = 6

    # Segment the volume from all three directions
    seg_z = three_class_segmentation_volume(
        inputImage=inputImage, direction="z", network=network
    )

    iteration = np.floor((100 * (1 / steps)))
    progress_bar.progress(int(iteration))

    seg_y = three_class_segmentation_volume(
        inputImage=inputImage, direction="y", network=network
    )

    iteration = np.floor((100 * (2 / steps)))
    progress_bar.progress(int(iteration))

    seg_x = three_class_segmentation_volume(
        inputImage=inputImage, direction="x", network=network
    )

    iteration = np.floor((100 * (3 / steps)))
    progress_bar.progress(int(iteration))

    # Rescale them to prevent overflow when we combine
    seg_z = rescale_16(seg_z)
    seg_y = rescale_16(seg_y)
    seg_z = combine_images(seg_z, seg_y)

    iteration = np.floor((100 * (4 / steps)))
    progress_bar.progress(int(iteration))

    # Free up memory
    seg_y = 0
    seg_x = rescale_16(seg_x)

    seg_z = combine_images(seg_z, seg_x)

    iteration = np.floor((100 * (5 / steps)))
    progress_bar.progress(int(iteration))

    seg_x = 0
    # Get the final product
    seg_z.CopyInformation(inputImage)
    seg = rescale_8(seg_z, verbose=False)

    iteration = np.floor((100 * (6 / steps)))
    progress_bar.progress(int(iteration))

    return seg


###
#  Classes for hacking the streamlit internals
#
#  !!!!If you touch this stuff everything will break!!!!
#
#
# Modifed from https://gist.github.com/Ghasel/0aba4869ba6fdc8d49132e6974e2e662
####


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
        
        st.experimental_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                self._state["data"], None
            ):
                self._state["is_rerun"] = True
                st.experimental_rerun()
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
