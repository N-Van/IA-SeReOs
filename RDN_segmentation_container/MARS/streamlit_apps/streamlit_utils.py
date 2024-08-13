import os
import sys
import glob
import pathlib
import pandas as pd
from PIL import Image
import getpass

import streamlit as st

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


def _direction_index(slice_direction="z"):
    slice_direction = str(slice_direction).lower()
    if slice_direction == "z":
        return int(2)
    elif slice_direction == "y":
        return int(1)
    else:
        return int(0)

supported_file_types = ["mhd", "nii", "tif", "png", "jpg", "bmp", "dcm"]
slice_types = ["tif", "png", "jpg", "bmp", "dcm"]
volume_types = ["mhd", "mha", "nii", "vtk"]

@st.cache
def _load_MARS_logo():
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    logo = Image.open(str(script_dir.joinpath('Moon_Logo_small.png')))
    return logo

def _get_tab_logo():
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    tab_logo = Image.open(str(script_dir.joinpath('moon_square.png')))
    return tab_logo

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

def _check_for_slice_to_vol(state, input_type, out_type, slice_types, volume_types, verbose=False):
    if input_type in slice_types and out_type in volume_types:
        if verbose == True:
            st.info(f"Slice input type: {input_type} and volume output type: {out_type} selected.")
            st.info(f" This is assumed to be an image stack and the slices will be converted to a volume output.")
        state.twoD_to_threeD = True
        current_user = _get_user()
        save_state_values(state=state, user=current_user)
    else:
        state.twoD_to_threeD = False


def _get_user():
    """
    Internal function to get the username for saving the settings.
    :return: Returns the operating system username using os.environ
    """
    #We can't always count on this being launchde from a C: on windows
    #So we get the current working directory, then if there are back slashes we grab the root drive letter.
    # current = pathlib.Path.cwd()
    # if "\\" in str(current):
    #     windows_drive = str(current.parts[0])
    # user = [os.environ["USERNAME"] if str(windows_drive) in os.getcwd() else os.environ["USER"]]
    return getpass.getuser()


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

def read_parameter_file(parm_file):
    parms = pd.read_csv(parm_file, index_col=0, comment="#")
    return parms

def save_state_values(state, user, app_name="RDN"):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(f"{user}_{app_name}_saved_state.json")
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


def load_state_values(state, user):
    script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    saved_dir = script_dir.joinpath("saved_states").joinpath(f"{user}_RDN_saved_state.json")
    restored_session = pd.read_json(str(saved_dir))
    st.write("Restored previous settings")
    state.model_path = str(restored_session['model_path'][0])
    state.model = restored_session['model'][0]
    state.input_path = restored_session['input_path'][0]
    state.input_type = restored_session['input_type'][0]
    state.output_path = pathlib.Path(restored_session['output_path'][0])
    state.out_type = restored_session['out_type'][0]
    state.parm_file = restored_session['parm_file'][0]
    return state

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

def gather_image_files(input_path, input_type):
    image_files = glob.glob(str(pathlib.Path(input_path).joinpath(f"*.{input_type}")))
    image_files.sort(key=natural_keys)
    st.write(f"Found {len(image_files)} image files for segmentation in {input_path}...")
    return image_files

def parm_from_directory(input_dir, input_file_type="tif", output_dir="", output_file_type="mhd"):
    # Directory to be scanned
    input_dir = pathlib.Path(input_dir)
    if input_file_type in slice_types:
        dir_obj = os.scandir(input_dir)
        dir_list = [str(input_dir.joinpath(entry.name)) for entry in dir_obj if entry.is_dir]
        parm_file = pd.DataFrame(dir_list)
        parm_file.columns = ["input_path"]
    else:
        dir_list = glob.glob(str(input_dir.joinpath(f"*.{input_file_type}").as_posix()))
        parm_file = pd.DataFrame(dir_list)
        parm_file[0] = parm_file[0].str.replace("\\", "/")
        parm_file = parm_file[0].str.rsplit("/", n=1, expand=True)
        parm_file.columns = ["input_path", "input_name"]
    parm_file["input_type"] = str(input_file_type)
    if output_dir == "":
        parm_file["output_path"] = parm_file["input_path"].astype(str).map('{}_seg'.format)
    else:
        parm_file["output_path"] = (output_dir)
    parm_file["output_type"] = str(output_file_type)
    return parm_file

def parm_from_par(par_file, out_type="mhd", input_type="mhd"):
    parm_file = pd.read_csv(par_file, sep=";")
    parm_file = parm_file[["$path", "$oldname"]]
    parm_file["$oldname"] = parm_file["$oldname"].str.replace("#", "")
    parm_file.columns = ["input_path", "input_name"]
    parm_file["input_path"] = parm_file["input_path"].map(lambda x: str(pathlib.Path(x).as_posix()))
    parm_file["input_path"] = parm_file["input_path"].str.cat("/" + parm_file["input_name"])
    parm_file["input_type"] = str(input_type)
    parm_file["output_path"] = parm_file["input_path"].astype(str).map('{}/01_Seg'.format)
    parm_file["input_path"] = parm_file["input_path"].astype(str).map('{}/00_Original'.format)
    parm_file["input_name"] = parm_file["input_name"].str.cat("." + parm_file["input_type"])
    parm_file["output_type"] = str(out_type)
    return parm_file
