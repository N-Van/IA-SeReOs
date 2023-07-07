import os
import re
import sys
import glob
import pathlib
import subprocess
import numpy as np
import pandas as pd

sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
from MARS.morphology.vtk_mesh import *
#from MARS.utils.readPar import *


inputdir = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation\Par")
os.chdir(inputdir)

# Read in the first par file. The 0 makes it the first one in the list.
#par_name = glob.glob("Black_earth_RDN_talus.csv")[0]
# par_name = glob.glob("RDN_done.par")[0]
par_name = glob.glob("Still_to_RDN.par")[0]

# Read in the paramter file and skip commented out lines
par_file = pd.read_csv(par_name, sep=";", comment="#")
print(par_file.columns)

# Place the parameter file into a new dataframe and replace the wildcard ($)
df = par_file
df.columns = df.columns.str.replace("$", "")
print(df.columns)

for row in df.iloc[:].itertuples():
    dir_out = pathlib.Path(str(row.folder)).joinpath(str(row.oldname)).joinpath("01_Seg")
    #dir_out = pathlib.Path(str(row.output_path))
    mhd_file = f"{str(row.oldname)}_RDN_seg.mhd"
    #mhd_file = f"{str(row.input_name).rpartition('.')[0]}_RDN_seg.mhd"
    seg_name = dir_out.joinpath(mhd_file)
    mid_plane_name = dir_out.joinpath(f'{mhd_file.replace(".mhd", "")}')
    print(dir_out)
    if seg_name.exists():
        print("YEEESSSSS!")
        df.oldname.loc[row.Index] = f"# {row.oldname}"
        # sitk_image = read_image(seg_name)
        # write_midplanes(inputImage=sitk_image, file_name=str(mid_plane_name))
        # seg = crude_threshold(sitk_image, threshold=128)
        # seg = rescale_intensity(seg, 0, 1, 0, 1)
        # seg = rescale_intensity(seg, 0, 1, 0, 255)
        # write_midplanes(inputImage=seg, file_name=f"{mid_plane_name}_128_threshold")

        # seg = crude_threshold(sitk_image, threshold=200)
        # seg = rescale_intensity(seg, 0, 1, 0, 1)
        # seg = rescale_intensity(seg, 0, 1, 0, 255)
        # write_midplanes(inputImage=seg, file_name=f"{mid_plane_name}_200_threshold")

        # seg = crude_threshold(sitk_image, threshold=254)
        # seg = rescale_intensity(seg, 0, 1, 0, 1)
        # seg = rescale_intensity(seg, 0, 1, 0, 255)
        # write_midplanes(inputImage=seg, file_name=f"{mid_plane_name}_254_threshold")
    else:
        print("NOOOPE")
        # df.oldname.loc[row.Index] = f"# {row.oldname}"

df = df[~df.oldname.str.contains("#")]

#Check to see if the original file exists
for row in df.iloc[:].itertuples():
    dir_out = pathlib.Path(str(row.folder)).joinpath(str(row.oldname)).joinpath("00_Original")    
    mhd_file = f"{str(row.oldname)}.mhd"
    mhd_file = dir_out.joinpath(mhd_file)
    if not mhd_file.exists():
        print("Nope!")
    
df.reset_index(drop=True, inplace=True)
df.to_csv(str(inputdir.joinpath(f"Still_to_RDN_May_2021.csv")), index=False, sep=";")

remove_list = ["ShaftDist", "ShaftMid", "ShaftProx", "Overview"]

omit = df[
    (df["portion"] == "Overview")
    | (df["portion"] == "ShaftDist")
    | (df["portion"] == "ShaftMid")
    | (df["portion"] == "ShaftProx")
    | (df["portion"] == nan)
]
df = df.drop(list(omit.index))

df.to_csv("RDN_done.par", sep=";", index=False)

for row in df.iloc[:].itertuples():
    dir_out = (
        pathlib.Path(str(row.folder)).joinpath(str(row.oldname)).joinpath("00_Original")
    )
    mhd_file = f"{str(row.oldname)}.mhd"
    mhd_name = dir_out.joinpath(mhd_file)
    # mid_plane_name = dir_out.joinpath(f'{mhd_file.replace(".mhd", "")}')
    print(dir_out)
    if mhd_name.exists():
        print("YEEESSSSS!")
        # sitk_image = read_image(seg_name)
        # write_midplanes(inputImage=sitk_image, file_name=str(mid_plane_name))
    else:
        print("NOOOPE")
        df.oldname.loc[row.Index] = f"# {row.oldname}"

df.to_csv("Still_to_RDN.par", sep=";", index=False)

par_file = glob.glob("Still_to_RDN.par")[0]

# Read in the paramter file and skip commented out lines
par_file = pd.read_csv(par_file, sep=";", comment="#")
print(par_file.columns)

# Place the parameter file into a new dataframe and replace the wildcard ($)
df = par_file
df.columns = df.columns.str.replace("$", "")
print(df.columns)
df.to_csv("Still_to_RDN.par", sep=";", index=False)
