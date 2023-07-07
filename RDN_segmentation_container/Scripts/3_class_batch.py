import os
import sys
import glob
import torch
import socket
import pathlib
import platform
import pandas as pd
from PIL import UnidentifiedImageError
from tqdm import tqdm

# Provide the location of the net folder. This will work until packaged.
sys.path.append(r"D:\Desktop\git_repo\MARS\morphology\segmentation\pytorch_segmentation")

if platform.system() == "Windows":
    if socket.gethostname() == 'L2ANTH-WT0023':
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\morphology\segmentation\pytorch_segmentation")
    else:
        sys.path.append(r"D:\Desktop\git_repo")
if platform.system().lower() == 'linux':
    if 'redhat' in platform.platform():
        sys.path.append(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/NStephens/git_repo")
    else:
        sys.path.append(r"/mnt/ics/RyanLab/Projects/NStephens/git_repo")

from MARS.morphology.segmentation.pytorch_segmentation.net.unet_light_rdn import UNet_Light_RDN
from MARS.morphology.segmentation.pytorch_segmentation.execute_3_class_seg import *

###################################################
#                                                 #
#                                                 #
#    Information to modify for this to work       #
#                                                 #
#                                                 #
###################################################

# The path where the model lives
model_path = pathlib.Path(r"C:\Users\nbs49\Desktop\pytorch_segmentation\models\model_6.pth")


par_file = pd.read_csv(r"Z:\RyanLab\Projects\nsf_human_variation\Par\Canids_Hum _windows.par", sep=";", comment="#")
df = par_file
df.columns = [header.replace("$", "") for header in df.columns]

# The path where the data is located
#To get a list of folders that you can then loop through
data_folder = pathlib.Path(r"Z:\RyanLab\Projects\For_Adam_G\Reoriented\global\spheres_15")
os.chdir(data_folder)

# the output folder for the segmentations
save_folder = data_folder.joinpath("3_class_seg")

###################################################
#                                                 #
#                                                 #
#  Shouldn't need to modify anything below this   #
#                                                 #
#                                                 #
###################################################

#Basic diagnostic information, which GPU device is in use, how many are available to use, and the name of the device
#torch.cuda.set_device(1)
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(device=None)


# Initiate the model
#Set up the UNet with 1 channel and 3 classes (air, dirt, bone).
net = UNet_Light_RDN(n_channels=1, n_classes=3)

# Load in the trained model
net.load_state_dict(torch.load(model_path))
net.cuda()
net.eval()

# Perform the segmentation
#three_class_segmentation(inDir=data_folder, outDir=save_folder, outType="tif")

# Perform segmentations on mhd files
failed = pd.DataFrame()

mhd_list = glob.glob("*.mhd")
mhd_list.sort()
mhd_total = len(mhd_list)

current_count = mhd_total
data_folder = pathlib.Path.cwd()
save_folder = data_folder.joinpath("3_class_segs")

for volume in enumerate(mhd_list[1:]):
    sitk_image = read_image(volume[1])
    border = thresh_simple(inputImage=sitk_image, background=254, foreground=255, outside=0)
    sitk_image = subtract_images(inputImage1=sitk_image, inputImage2=border)
    out_name = f"{volume[1][:-4]}_pytorch_seg"
    try:
        seg = three_class_seg_xyz(inputImage=sitk_image, network=net)
        seg.CopyInformation(sitk_image)
        write_image(inputImage=seg, outName=out_name, outDir=save_folder, fileFormat="mhd")
    except:
        last_file = out_name
        last_file = pd.DataFrame({"Restart_from": last_file[-1]}, index=[volume[0]])
        print(f"{last_file} failed to segment...")
        failed = pd.concat([failed, last_file], axis=0)
    finally:
        current_count -= 1
        print(f"\n{current_count} out of {mhd_total} volumes remaining to process...\n")

failed.index.name = ["List_position"]
failed.to_csv("Failed_segmentations.csv")



#The par file you want to work with
par_file = pd.read_csv(r"Z:\RyanLab\Projects\nsf_human_variation\Par\Canids_Hum _windows.par", sep=";")#, comment="#")

#Change it to the short hand for dataframe
df = par_file


#Replace the dollar signs in the header
df.columns = [header.replace("$", "") for header in df.columns]
df["oldname"] = df["oldname"].str.replace("#", "")

#Switch to the base directory where the failed par file will be written to
working_dir = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation")
os.chdir(working_dir)

#Get the length of the items to process
current_count = len(df)
current_total = current_count

failed = pd.DataFrame()
#Loop through the rows one by one
for row in df.itertuples():

     #Get the input name for the row from the oldname (i.e. the unique identifier)
    inName = str(row.oldname)

    #Define the output name by the input name
    out_name = f"{inName}_pytorch_seg"

    # Get the input folder from the folder column, where multiple skeeltal elements are for each specimen
    inDir = str(row.folder)

    # Define the imageFolder by the oldname, which contains the unique identifier
    imageFolder = pathlib.Path(inDir).joinpath(inName)

    # Helpful print message
    print(f"\nWorking on {inName}...\n")

    # The input data is in the medtool folder structure style so 00_original is where the unseg data lives.
    data_folder = imageFolder.joinpath("00_Original")

    # The output folder is the segmentation folder
    save_folder = imageFolder.joinpath("01_Seg")

    # Read in the unsegmented image from the data folder
    sitk_image = read_image(str(data_folder.joinpath(f"{inName}.mhd")))

    #Use a try statement so if something goes wrong it will just go on to the except statement
    try:
        #Segment in 2d all three primary planes (x, y, z)
        seg = three_class_seg_xyz(inputImage=sitk_image, network=net)

        #Copy the metadata over from the unsegmented data.
        seg.CopyInformation(sitk_image)

        # Write the volume out to theoutput folder
        write_image(inputImage=seg, outName=out_name, outDir=save_folder, fileFormat="mhd")

    # If something goes wrong in the try statement, the except statement should do this
    except:

        # Make a dataframe out of the row, print a statement, and then append it to the failed dataframe.
        last_file = pd.DataFrame(row)
        failed = pd.concat([failed, last_file], axis=0)

        # Let it print the console so we know if we happen to be watching.
        print(f"{inName} failed to segment...")
    # The finally statement will exectute the code no matter what, so we simply fill it with the things we want to happen.


failed.to_csv("Failed_pytorch_segmentations.par", sep=";")



###DICOMS

directory = pathlib.Path(r"D:\Desktop\Propithecus_TIFFS")
os.chdir(directory)
tif_stack = glob.glob("*.tif")

dicom_dir = directory.joinpath("dicom")

#sitk_image = read_stack(tif_stack)
sitk_image, metadata = read_dicom(tif_stack)

res = sitk_image.GetSpacing()
res = (0.1177, 0.1177, 0.1177)
sitk_image.SetSpacing(res)

sitk_image = rescale_8(sitk_image)

write_dicom(inputImage=sitk_image, metadata=metadata, outName="Propithecus", outDir=dicom_dir)

dicom_stack = glob.glob(str(dicom_dir.joinpath("*.dcm")))
sitk_image, metadata = read_dicom(inputStack=dicom_stack)

seg.CopyInformation(sitk_image)
write_image(inputImage=seg, outName="Propithecus", outDir=directory, fileFormat="mhd")

"""
#Check to see if you can read nsipro with something along these lines
#https://docs.python.org/3/library/struct.html#struct.calcsize
import struct
with open(amira_file, "rb") as binary_file:
    couple_bytes = binary_file.read(int(400))
    for i in range(400):
        binary_file.seek(i)
        print(struct.unpack("<c", couple_bytes[i:i+1]))

"""


