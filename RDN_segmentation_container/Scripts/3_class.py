import os
import sys
import socket
import pathlib
import platform
import tempfile
from subprocess import PIPE, Popen

if platform.system() == "Windows":
    if socket.gethostname() == 'L2ANTH-WT0023':
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
    else:
        sys.path.append(r"D:\Desktop\git_repo")
if platform.system().lower() == 'linux':
    if 'redhat' in platform.platform():
        sys.path.append(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/NStephens/git_repo")
    else:
        sys.path.append(r"/mnt/ics/RyanLab/Projects/NStephens/git_repo")

from MARS.utils.check_environments import check_environment_location, write_temp_bat_windows

#Check if pytorch environment is available, and if so get it's location.
pytorch_env = check_environment_location(env_name="pytorch_seg")

#This should work after packaged
#gui_path = pathlib.Path(os.path.abspath(os.path.dirname(sys.argv[0]))).parent.parent.joinpath("morphology").joinpath("segmentation").joinpath("pytorch_segmentation")


if isinstance(pytorch_env, bool):
    print(f"Pytorch environment not found!")
else:
    pytorch_env = pytorch_env["Location"][0]
    pytorch_python = pathlib.Path(pytorch_env).joinpath("python")
    three_class_script = pathlib.Path(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\morphology\segmentation\pytorch_segmentation\3_class_gui.py")
    temp_batch = write_temp_bat_windows(batch_name="3_class", python_location=pytorch_python, script_location=three_class_script)
    process = Popen(f"{temp_batch}", shell=True, stdin=PIPE, stdout=PIPE)

