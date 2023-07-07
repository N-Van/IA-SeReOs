import os
import glob
import pathlib
import subprocess

directory = r"Z:\RyanLab\Projects\CT_Data\Belize\Vols\ST_18_14_Check"
os.chdir(directory)

vol_list = pathlib.Path.rglob(pathlib.Path.cwd(), pattern="*.vol")
vol_list = [x for x in vol_list]

meta_script = r"Z:\RyanLab\Projects\NStephens\git_repo\medtool_plugins\CT_scan_metadata_readers\Ct_metadata_reader.py"
mesh_script = r"Z:\RyanLab\Projects\AGuerra\Scripts\mesh_reoriented.py"

for vol in vol_list[:]:
    directory = str(vol.parent.as_posix())
    file_name = str(vol.as_posix())
    par_oldname = str(vol.name).replace(".vol", "")
    par_name = par_oldname
    text_match = str("''")

    # command = f"python {meta_script} {directory} {file_name} {par_oldname} {par_name} {text_match}"
    # print(command)
    # executed = subprocess.Popen(f"{command}", shell=True)
    # out, err = executed.communicate()
    # print(out)
    ply_file = glob.glob(str(pathlib.Path(directory).joinpath("*.ply")))
    if ply_file:
        if pathlib.Path(ply_file[0]).exists():
            print(ply_file)
    else:
        command = f"python {mesh_script} {directory}"
        executed = subprocess.Popen(f"{command}", shell=True)
        out, err = executed.communicate()
        print(out)


import os
import re
import sys
import glob
import shutil
import pathlib
import subprocess
import numpy as np
import pandas as pd
import SimpleITK as sitk
from timeit import default_timer as timer
from typing import Union, Any, List, Optional, cast

sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
from MARS.utils.readPar import *
from MARS.morphology.vtk_mesh import *


def read_mhd(directory, mhd_file):
    """
    If there are no metadata files, and there are diccoms this function will try to extract hat information.
    :param directory:
    :return:
    """
    original_directory = pathlib.Path.cwd()
    directory = pathlib.Path(directory)
    os.chdir(directory)
    mhd_file = str(directory.joinpath(mhd_file))
    print(f"Reading {mhd_file}...")
    reader = sitk.ImageFileReader()
    reader.SetFileName(mhd_file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    xdim, ydim, zdim = reader.GetSize()
    xres, yres, zres = reader.GetSpacing()
    print(f"Image Size: {xdim} {ydim} {zdim}")
    print(f"Image Resolution: {xres} {yres} {zres}")
    if xres == 1.0:
        print(
            f"\n\n\n!!!!Image spacing is set as 1.0 which is unlikely to be true. \n"
            f"ImageJ often writes out mhd files in this way, ignoring spacing. Please manually correct this file.!!!!!\n\n\n"
        )
    if xres != yres:
        res = [xres, yres, zres]
        print(f"Resolution: {xres, yres, zres}")
    elif xres != zres:
        res = [xres, yres, zres]
        print(f"Resolution: {xres, yres, zres}")
    elif yres != zres:
        res = [xres, yres, zres]
        print(f"Resolution: {xres, yres, zres}")
    else:
        res = xres
        print(f"Resolution: {res}")
    os.chdir(original_directory)
    return xdim, ydim, zdim, res


def ct_log_reader(directory, log_file):
    """
    This reads in a skyscan file and extracts the spacing (resolution)
    :param directory: A pathlib formated directory or a directory in string format (e.g. r"Z:/RyanLab").
    :param pcr_file: A pcr file with the same name as the vol file.
    :return: Returns the x,y,z dimensions and resolution.
    """
    print(log_file)
    if isinstance(log_file, list):
        log_file = log_file[0]
    directory = pathlib.Path(directory)
    file_name = str(log_file)
    log_file = pathlib.Path(directory).joinpath(file_name)

    # Reads in the file
    print("\nOpening log file....\n")
    # Reads in the file to find the line number for the resolution
    with open(str(log_file), "rt") as in_file:

        # Search for the resolution using a string
        searchres = "Image Pixel Size (um)="
        print(searchres)

        # Then create content object and read the file line by line
        content = in_file.readlines()

        # Read the resolution line
        res_index = [x for x in range(len(content)) if str(searchres) in content[x]]
        res_index = int(res_index[0])
        print(f"Resolution found on line {res_index}")
        res = content[res_index].strip()
        res = res.replace(f"{searchres}", "")
        print(f"Resolution is {res}")

        # So we close the file so it isn't in memory anymore.
        in_file.close()
        res = float(res) * 0.001

    print(f"Resolution: {res}")
    return res


def write_meta_header(filename, meta_dict):
    header = ""
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = [
        "ObjectType",
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "CompressedData",
        "CompressedDataSize",
        "TransformMatrix",
        "Offset",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "DimSize",
        "ElementType",
        "ElementDataFile",
        "Comment",
        "ElementNumberOfChannels",
        "SeriesDescription",
        "AcquisitionDate",
        "AcquisitionTime",
        "StudyDate",
        "StudyTime",
    ]
    for tag in tags:
        if tag in meta_dict.keys():
            header += "%s = %s\n" % (tag, meta_dict[tag])
    f = open(filename, "w")
    f.write(header)
    f.close()


def write_mhd_file_vol(
    mhdfile: str, spacing: float, dsize: list, bits: str = "MET_FLOAT"
):
    assert mhdfile[-4:] == ".mhd"
    meta_dict = {}
    meta_dict["ObjectType"] = "Image"
    meta_dict["BinaryData"] = "True"
    meta_dict["BinaryDataByteOrderMSB"] = "False"
    meta_dict["CompressedData"] = False
    meta_dict["TransformMatrix"] = "1 0 0 0 1 0 0 0 1"
    meta_dict["Offset"] = "0 0 0"
    meta_dict["CenterOfRotation"] = "0 0 0"
    meta_dict["AnatomicalOrientation"] = "RAI"
    meta_dict["ElementType"] = str(bits)
    meta_dict["ElementNumberOfChannels"] = 3
    meta_dict["ElementSpacing"] = f"{float(spacing)} {float(spacing)} {float(spacing)}"
    meta_dict["NDims"] = "3"
    meta_dict["DimSize"] = f"{int(dsize[0])} {int(dsize[1])} {int(dsize[2])}"
    meta_dict["ElementDataFile"] = os.path.split(mhdfile)[1].replace(".mhd", ".raw")
    write_meta_header(mhdfile, meta_dict)
    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + "/" + meta_dict["ElementDataFile"]
    else:
        data_file = meta_dict["ElementDataFile"]
    shutil.move(
        pathlib.Path.cwd().joinpath(mhdfile), pathlib.Path.cwd().joinpath(mhdfile)
    )


########################################
#                                      #
# This is where we actually do stuff   #
#                                      #
########################################

# initial_path = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation")
# initial_path = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation")

# Define the directory of the paremter file and change to it
inputdir = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation\Par")
os.chdir(inputdir)

# Read in the first par file. The 0 makes it the first one in the list.
par_file = glob.glob("Black_earth_RDN_talus.csv")[0]

population = "TX_State"

# Read in the paramter file and skip commented out lines
par_file = pd.read_csv(par_file, sep=",", comment="#", index_col=0)
print(par_file.columns)

# Place the parameter file into a new dataframe and replace the wildcard ($)
df = par_file
df.columns = df.columns.str.replace("$", "")
print(df.columns)

out_dir = pathlib.Path(r"Z:\RyanLab\Projects\NStephens\SSRI CT Scans\Volumes")
os.chdir(out_dir)

for row in df.iloc[:].itertuples():
    input_location = pathlib.Path(row.location)
    file_name = row.file
    out_name = row.oldname
    # if not out_dir.joinpath(f"{out_name}_midplanes_midplanes.png").exists():
    file_type = ".bmp"
    remove_file = "_spr.bmp"
    remove_file_stack = glob.glob(str(input_location.joinpath(f"*{remove_file}")))
    if remove_file_stack:
        print("SPR is present")
        remove_file = "_spr.bmp"
        file_stack = glob.glob(str(input_location.joinpath(f"*{file_type}")))
        file_stack.sort()
        file_stack = [x for x in file_stack if remove_file not in x]
        input_image = sitk.ReadImage(file_stack, sitk.sitkUInt8)
        ct_log_dir = pathlib.Path(row.location)
        ct_log_dir = pathlib.Path(row.location)
        ct_log_file = glob.glob(str(ct_log_dir.joinpath("*_rec.log")))
        if len(ct_log_file) == 2:
            ct_log_file.sort(key=os.path.getmtime, reverse=True)
        ct_log_file = ct_log_file[-1]
        out_name = row.oldname
        correct_res = ct_log_reader(directory=ct_log_dir, log_file=ct_log_file)
        input_image.SetSpacing([correct_res, correct_res, correct_res])
        write_midplanes(input_image, file_name=f"{row.oldname}_midplanes")

        try:
            write_image(
                inputImage=input_image,
                outName=out_name,
                outDir=out_dir,
                fileFormat="mhd",
            )
        except:
            print(f"\n\n\n\n\n {out_name} didn't write out!\n\n\n\n\n ")


for row in df.iloc[:].itertuples():
    ct_log_dir = pathlib.Path(row.location)
    ct_log_file = glob.glob(str(ct_log_dir.joinpath("*_rec.log")))
    if len(ct_log_file) == 2:
        ct_log_file.sort(key=os.path.getmtime, reverse=True)
    ct_log_file = ct_log_file[-1]
    out_name = row.oldname
    correct_res = ct_log_reader(directory=ct_log_dir, log_file=ct_log_file)

    dim_x, dim_y, dim_z, res = read_mhd(
        directory=r"Z:\RyanLab\Projects\NStephens\SSRI CT Scans\Volumes",
        mhd_file=f"{out_name}.mhd",
    )

    write_mhd_file_vol(
        mhdfile=f"{out_name}.mhd",
        spacing=correct_res,
        dsize=[dim_x, dim_y, dim_z],
        bits="MET_UCHAR",
    )


directory = pathlib.Path(r"Z:\RyanLab\Projects\NStephens\SSRI CT Scans\Skull_499")
os.chdir(directory)

my_files = [x for x in directory.rglob("*Skull_499_rec_ (1).bmp")]


# If you wnat to subset this, then you can use iloc, which goes [row,row, column:column]
for row in df.iloc[:1].itertuples():
    # Setup the image input
    input_dir = pathlib.Path(row.input_path)
    input_name = str(row.input_name)

    in_file = pathlib.Path(input_dir).joinpath(input_name)
    # setup the image output
    output_name = input_name.replace(".mhd", "")

    if in_file.exists():
        sitk_image = read_image(in_file)

        if "Femur" in input_name:
            resampled_amount = 0.15
        elif "Tibia" in input_name:
            resampled_amount = 0.15
        elif "Humerus" in input_name:
            resampled_amount = 0.1
        else:
            resampled_amount = 0.1
        resampled = resample_sitk_image(
            sitk_image,
            spacing=float(resampled_amount),
            interpolator="linear",
            fill_value=0,
        )
        seg = thresh_otsu_multi(inputImage=resampled, groups=3, bins=256, threads=6)
        seg = crude_threshold(inputImage=seg, threshold=1)
        # write_midplanes(
        #     inputImage=seg,
        #     file_name=f"{str(in_file).replace('.mhd', '')}",
        #     title="",
        #     margin=0,
        #     dpi=100,
        #     fig_scale=1,
        # )

        seg = rescale_intensity(
            inputImage=seg, old_min=0, old_max=1, new_min=0, new_max=1, threads=6
        )

        # Do a quick closing operation to limit the amount of floating bits for the connected components filter to consider.
        closed = closing_morph(inputImage=seg, closing_kernel=5, threads=6)

        # Resample the resample a bit further so we can rapidly open it later in paraview.
        # resampled = resample_sitk_image(resampled, spacing=0.3, interpolator="linear", fill_value=0)
        write_image(
            closed,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            fileFormat="mhd",
        )
        resampled_mhd_name = input_dir.joinpath(f"{output_name}_resampled_seg.mhd")
        # Read the mhd into vtk, so it is in the proper data format
        vtk_image = vtk_read_mhd(str(resampled_mhd_name))

        # Mesh with VTK by reading in the image and then using their marching cubes algorithm
        mesh = vtk_MarchingCubes(vtk_image, threshold=1, extract_largest=True)

        # Write out the ply as a binary
        vtk_writePLY(
            mesh,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            outType="binary",
        )

##############################################################
#                                                            #
#  Volume to mesh with crop by mesh with Medtool par file    #
#                                                            #
##############################################################

for row in df.iloc[1:].itertuples():
    # Setup the image input
    input_dir = pathlib.Path(row.path).joinpath(row.oldname).joinpath("01_Seg")
    input_name = f"{str(row.name)}_seg.mhd"
    in_file = pathlib.Path(input_dir).joinpath(input_name)
    # setup the image output
    output_name = input_name.replace(".mhd", "")

    if in_file.exists():
        sitk_image = read_image(in_file)

        if "Femur" in input_name:
            resampled_amount = 0.15
        elif "Tibia" in input_name:
            resampled_amount = 0.15
        elif "Humerus" in input_name:
            resampled_amount = 0.1
        else:
            resampled_amount = 0.1
        resampled = resample_sitk_image(
            sitk_image,
            spacing=float(resampled_amount),
            interpolator="linear",
            fill_value=0,
        )
        seg = thresh_otsu_multi(inputImage=resampled, groups=3, bins=256, threads=6)
        seg = crude_threshold(inputImage=seg, threshold=1)
        seg = rescale_intensity(
            inputImage=seg, old_min=0, old_max=1, new_min=0, new_max=1, threads=6
        )

        # Do a quick closing operation to limit the amount of floating bits for the connected components filter to consider.
        closed = closing_morph(inputImage=seg, closing_kernel=5, threads=6)

        # Resample the resample a bit further so we can rapidly open it later in paraview.
        # resampled = resample_sitk_image(resampled, spacing=0.3, interpolator="linear", fill_value=0)
        write_image(
            closed,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            fileFormat="mhd",
        )
        resampled_mhd_name = input_dir.joinpath(f"{output_name}_resampled_seg.mhd")
        # Read the mhd into vtk, so it is in the proper data format
        vtk_image = vtk_read_mhd(str(resampled_mhd_name))

        # Mesh with VTK by reading in the image and then using their marching cubes algorithm
        mesh = vtk_MarchingCubes(vtk_image, threshold=1, extract_largest=True)

        # Write out the ply as a binary
        vtk_writePLY(
            mesh,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            outType="binary",
        )

        mhd_file = f"{row.name}_seg.mhd"
        mhd_file = input_dir.joinpath(f"{mhd_file}")  # Original mhd
        cropped_name = mhd_file.name.replace(".mhd", "_cropped")  # The output name
        mesh_file = input_dir.joinpath(f"{output_name}_resampled_seg.ply")
        sitk_image = read_image(inputImage=mhd_file)  # Read it in
        res = sitk_image.GetSpacing()[0]  # Get the resolution from the scan

        # Get the bounds from the mesh
        mesh_coords = bounds_from_mesh(
            mesh_file=mesh_file, padding=10.0, image_resolution=res
        )
        # Crop the image volume
        sitk_cropped = crop_image(
            inputImage=sitk_image,
            crop_amount=0,
            crop_dims=mesh_coords,
            crop_unit="physical",
            resolution="",
            keep_square="no",
        )

        # Write out the midplanes for quick verification
        write_midplanes(
            sitk_cropped,
            file_name=str(input_dir.joinpath(cropped_name.replace(".mhd", ""))),
        )

        # Write out the cropped image volume
        write_image(
            sitk_cropped, outName=cropped_name, outDir=input_dir, fileFormat="mhd"
        )


##############################################################
#                                                            #
#  Volume to mesh with crop by mesh with RDN batch file      #
#                                                            #
##############################################################

df = pd.read_csv(
    r"Z:\RyanLab\Projects\nsf_human_variation\Par\Black_earth_RDN_talus.csv",
    comment="#",
    index_col=0,
)
df.input_name = df.input_name.str.replace("_cropped", "")

for row in df.iloc[:1].itertuples():
    # Setup the image input
    input_dir = pathlib.Path(row.input_path)
    input_name = f"{str(row.input_name)}"
    in_file = pathlib.Path(input_dir).joinpath(input_name)
    # setup the image output
    output_name = input_name.replace(".mhd", "")

    if in_file.exists():
        # time.sleep(10)
        sitk_image = read_image(in_file)
        get_mean = sitk.StatisticsImageFilter()
        get_mean.Execute(sitk_image)
        mean_value = get_mean.GetMean()
        if mean_value > 100:
            shift_value = float(-1 * mean_value) + get_mean.GetVariance()  # * 0.5)
            # Rescale the inensity by shifting the scale to the left to destecko it.
            sitk_image = sitk.ShiftScale(sitk_image, shift_value)

        sitk_image = sitk.Cast(
            sitk.RescaleIntensity(sitk_image, outputMinimum=0, outputMaximum=255),
            sitk.sitkUInt8,
        )

        write_midplanes(sitk_image, file_name=str(input_dir.joinpath(output_name)))

        if "Femur" in input_name:
            resampled_amount = 0.15
        elif "Tibia" in input_name:
            resampled_amount = 0.15
        elif "Humerus" in input_name:
            resampled_amount = 0.1
        else:
            resampled_amount = 0.1

        resampled = resample_sitk_image(
            sitk_image,
            spacing=float(resampled_amount),
            interpolator="linear",
            fill_value=0,
        )

        seg = thresh_otsu_multi(inputImage=resampled, groups=3, bins=256, threads=6)
        seg = crude_threshold(inputImage=seg, threshold=1)
        seg = rescale_intensity(
            inputImage=seg, old_min=0, old_max=1, new_min=0, new_max=1, threads=6
        )

        # Do a quick closing operation to limit the amount of floating bits for the connected components filter to consider.
        closed = closing_morph(inputImage=seg, closing_kernel=5, threads=6)

        # Resample the resample a bit further so we can rapidly open it later in paraview.
        # resampled = resample_sitk_image(resampled, spacing=0.3, interpolator="linear", fill_value=0)
        write_image(
            closed,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            fileFormat="mhd",
        )
        resampled_mhd_name = input_dir.joinpath(f"{output_name}_resampled_seg.mhd")
        # Read the mhd into vtk, so it is in the proper data format
        vtk_image = vtk_read_mhd(str(resampled_mhd_name))

        # Mesh with VTK by reading in the image and then using their marching cubes algorithm
        mesh = vtk_MarchingCubes(vtk_image, threshold=1, extract_largest=True)

        # Write out the ply as a binary
        vtk_writePLY(
            mesh,
            outName=f"{output_name}_resampled_seg",
            outDir=input_dir,
            outType="binary",
        )

        mhd_file = f"{row.input_name}"
        mhd_file = input_dir.joinpath(f"{mhd_file}")  # Original mhd
        cropped_name = mhd_file.name.replace(".mhd", "_cropped")  # The output name
        mesh_file = input_dir.joinpath(f"{output_name}_resampled_seg.ply")
        # sitk_image = read_image(inputImage=mhd_file)  # Read it in
        res = sitk_image.GetSpacing()[0]  # Get the resolution from the scan

        # Get the bounds from the mesh, the padding is in physical units (mm)
        mesh_coords = bounds_from_mesh(
            mesh_file=mesh_file, padding=20.0, image_resolution=res
        )
        # Crop the image volume
        sitk_cropped = crop_image(
            inputImage=sitk_image,
            crop_amount=0,
            crop_dims=mesh_coords,
            crop_unit="physical",
            resolution="",
            keep_square="no",
        )

        # Write out the midplanes for quick verification
        write_midplanes(
            sitk_cropped,
            file_name=str(input_dir.joinpath(cropped_name.replace(".mhd", ""))),
        )

        # Write out the cropped image volume
        write_image(
            sitk_cropped, outName=cropped_name, outDir=input_dir, fileFormat="mhd"
        )
