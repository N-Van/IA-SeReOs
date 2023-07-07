import os
import re
import sys
import math
import glob
import difflib
import pathlib
import tempfile
import multiprocessing
import numpy as np
import streamlit as st
import SimpleITK as sitk
from PIL import Image
from PIL import ImageOps
from timeit import default_timer as timer

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
        print(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        print(f"{message} took: {float(elapsed):10.4f} seconds")


def rescale_labels(inputFilename, writeOut=False):
    """
    Load in a 2d image file and rescale for label prep.
    :param inputFilename: Name of file to be rescaled. Can be anything that SimpleITK reads.
    :return: Returns a rescaled tif image.
    """

    inputImage = sitk.ReadImage(inputFilename)
    MinMax = sitk.MinimumMaximumImageFilter()
    MinMax.Execute(inputImage)

    pixelid = inputImage.GetPixelIDValue()
    if pixelid != 1:
        print(f"Rescaling to unsigned 8bit...")
        inputImage = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)

    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMinimum(0)
    imageMax = MinMax.GetMaximum()
    print(f"Image max intensity is {imageMax}...\n")

    if int(imageMax) == 1:
        rescaled = inputImage
    else:
        print(f"Rescaling max intensity to 1...\n")
        rescaleFilter.SetOutputMaximum(1)
        rescaled = rescaleFilter.Execute(inputImage)

    if writeOut != False:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(f"{inputFilename[:-4]}_rescaled.tif")
        writer.Execute(rescaled)
    else:
        return rescaled

def rescale_by_label(bone_label, dirt_label, check_label=False, expected_classes=3):
    dirt_label = subtract_images(inputImage1=bone_label, inputImage2=dirt_label)

    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMinimum(0)
    rescaleFilter.SetOutputMaximum(255)
    bone_rescaled = rescaleFilter.Execute(bone_label)

    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMinimum(0)
    rescaleFilter.SetOutputMaximum(128)
    dirt_rescaled = rescaleFilter.Execute(dirt_label)

    label = sitk.NaryAdd(bone_rescaled, dirt_rescaled)
    if check_label != False:
        _check_label(label, expected_classes=int(expected_classes))
    return label


def subtract_images(inputImage1, inputImage2):
    """
    Function to subtract two SimpleITK images using the Subtract filter.

    :param inputImage1: A SimpleITK image.
    :param inputImage2: A SimpleITK image.
    :return: Returns a single SimpleITK image.
    """
    start = timer()

    # Subtract the two images together
    print("Subtracting...")
    subtracted = sitk.Subtract(inputImage1, inputImage2)
    _end_timer(start, message="Subtracting the two images")
    return subtracted

def combine_images(inputImage1, inputImage2):
    """
    Function to combine two SimpleITK images using the Add filter.

    :param inputImage1: SimpleITK image.
    :param inputImage2: SimpleITK image.
    :return: Returns a single SimpleITK image.
    """
    start = timer()

    # Add the two images together
    print("Combining...")
    combined = sitk.Add(inputImage1, inputImage2)
    _end_timer(start, message="Combing the two images")
    return combined

def write_label(inputImage, out_name, file_type="tif", out_dir=""):
    if out_dir == "":
        out_dir = pathlib.Path.cwd()
    else:
        out_dir = pathlib.Path(out_dir)
    if "." in file_type:
        file_type = file_type.replace(".", "")
    if "\\" or "/" in str(out_name):
        out_name = pathlib.Path(out_name).parts[-1].split(".")[0]

    outName = out_dir.joinpath(f"{out_name}.{file_type}")
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(outName))
    writer.Execute(inputImage)

def _check_label(inputImage, expected_classes):
    nda = sitk.GetArrayFromImage(inputImage)
    num_classes, class_proportion = np.unique(nda, return_counts=True)
    print(num_classes)
    if len(num_classes) != expected_classes:
        print(f"There are {num_classes} instead of {expected_classes} classes in the label!")
        print(f"{class_proportion}")

def process_dragonfly_labels(labels_location, output_name, out_dir, extract_strings=["Air", "Dirt", "Bone"]):

    start = timer()
    labels = labels_location
    extract_strings.sort()
    out_dir = pathlib.Path(out_dir)

    labels = [image_file for string_match in extract_strings for image_file in labels if string_match in image_file]
    labels.sort()

    slice_num = str(pathlib.Path(labels[0]).parts[-1]).split(" ", 1)[0].replace(str(extract_strings[0]), " ")
    output_name = f"{output_name}{slice_num}"

    air_label, bone_label, dirt_label = [rescale_labels(inputFilename=label, writeOut=False) for label in labels]

    label = rescale_by_label(bone_label=bone_label, dirt_label=dirt_label, check_label=False, expected_classes=3)

    write_label(inputImage=label, out_name=f"{output_name}", file_type="png", out_dir=out_dir)
    print("\n")
    _end_timer(start_timer=start, message="Composing labels")

def rescale_intensity(inputFilename, writeOut=True, file_type="", outDir=""):
    """
    Load in a 2d image file and rescale for data augmentation.
    :param inputFilename: Name of file to be resclaed. Can be anything that SimpleITK reads.
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


    inputImage = sitk.ReadImage(inputFilename, sitk.sitkUInt8)
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

def downscale_intensity(inputFilename, downscale_value=100, writeOut=True, file_type="", outDir=""):
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


    inputImage = sitk.ReadImage(inputFilename, sitk.sitkUInt8)
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

def clean_image(inputFilename, suffix="", out_name="", out_type="", out_dir="", remove_space=True, remove_dblundr=True, to_streamlit=False):
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

        out_name = str(pathlib.Path(out_dir).joinpath(out_name))

    inputImage = sitk.ReadImage(inputFilename, sitk.sitkUInt8)
    if to_streamlit:
        st.write((f"\nWriting out {out_name} as {out_type}....\n"))
    else:
        print(f"\nWriting out {out_name} as {out_type}....\n")
    writer = sitk.ImageFileWriter()
    writer.SetFileName(f"{out_name}{suffix}.{out_type}")
    writer.Execute(inputImage)



def check_for_match(missing_file, check_filelist):
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

    if len(matches) != 0:
        pairs = [(missing_file, matched) for matched in matches]
        grey_text = []
        green_text = []
        red_text = []
        for checking, match_file in pairs:
            for i, s in enumerate(difflib.ndiff(checking, match_file)):
                if s[0] == ' ':
                    grey_text.append((i))
                elif s[0] == '-':
                    green_text.append(i)
                elif s[0] == '+':
                    red_text.append(i)
            print(f"\nDifferences between {checking}")
            print(f"                and {match_file}:")
            if len(green_text) > 0:
                print(f"Adds:")
                missing_g = [f"{checking[g_text]}" for g_text in green_text]
                [green_print(g) for g in missing_g]

            if len(red_text) > 0:
                print(f"Removes:")
                missing_r = [f"{match_file[r_text]}" for r_text in red_text]
                [red_print(r) for r in missing_r]
        return matches
    else:
        print(f"No matches for {checking} found =(")

def red_print(text):
    print(f'\033[31m{text}\033[0m', end="")

def green_print(text):
    print(f'\033[32m{text}\033[0m', end="")

def gray_print(text):
    print(f'\033[90m{text}\033[0m', sep="")

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
    print(f"file size: {file_s[0]}")
    size = int(np.ceil(file_s[1]))
    return size

def read_raw(binary_file_name, image_size, sitk_pixel_type, image_spacing=None,
             image_origin=None, big_endian=False):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {sitk.sitkUInt8: 'MET_UCHAR',
                  sitk.sitkInt8: 'MET_CHAR',
                  sitk.sitkUInt16: 'MET_USHORT',
                  sitk.sitkInt16: 'MET_SHORT',
                  sitk.sitkUInt32: 'MET_UINT',
                  sitk.sitkInt32: 'MET_INT',
                  sitk.sitkUInt64: 'MET_ULONG_LONG',
                  sitk.sitkInt64: 'MET_LONG_LONG',
                  sitk.sitkFloat32: 'MET_FLOAT',
                  sitk.sitkFloat64: 'MET_DOUBLE'}
    direction_cosine = ['1 0 0 1', '1 0 0 0 1 0 0 0 1',
                        '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1']
    dim = len(image_size)
    header = ['ObjectType = Image\n'.encode(),
              ('NDims = {0}\n'.format(dim)).encode(),
              ('DimSize = ' + ' '.join([str(v) for v in image_size]) + '\n')
              .encode(),
              ('ElementSpacing = ' + (' '.join([str(v) for v in image_spacing])
                                      if image_spacing else ' '.join(
                  ['1'] * dim)) + '\n').encode(),
              ('Offset = ' + (
                  ' '.join([str(v) for v in image_origin]) if image_origin
                  else ' '.join(['0'] * dim) + '\n')).encode(),
              ('TransformMatrix = ' + direction_cosine[dim - 2] + '\n')
              .encode(),
              ('ElementType = ' + pixel_dict[sitk_pixel_type] + '\n').encode(),
              'BinaryData = True\n'.encode(),
              ('BinaryDataByteOrderMSB = ' + str(big_endian) + '\n').encode(),
              # ElementDataFile must be the last entry in the header
              ('ElementDataFile = ' + os.path.abspath(
                  binary_file_name) + '\n').encode()]
    fp = tempfile.NamedTemporaryFile(suffix='.mhd', delete=False)

    [print(head.rstrip().decode()) for head in header]
    print("\n\n")
    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img

def get_amira_header(amira_file, search_length=2048):
    with open(amira_file, "rb") as binary_file:
        binary_file.read(int(search_length))
        num_bytes = binary_file.tell()  # Get the file size
        print(f"Searching for header in first {num_bytes} bytes...")
        for i in range(num_bytes):
            binary_file.seek(i)
            eight_bytes = binary_file.read(25)
            if eight_bytes == b'# Data section follows\n@1':
                print(f"Found header end at position {str(i)}!")
                header_end = i + 25
    return header_end

def read_amira_header(amira_file, header_length=514):
    with open(amira_file, "rb") as binary_file:
        # Seek a specific position in the file and read N bytes
        binary_file.seek(0, 0)  # Go to beginning of the file
        couple_bytes = binary_file.read(int(header_length))
        header = [c_bytes for c_bytes in couple_bytes.split(b'\n')]
    file_length_in_bytes = os.path.getsize(amira_file)
    binary_length = file_length_in_bytes - header_length
    bounds = []
    for items in enumerate(header):
        if "Content" in str(items):
            dims = [int(x) for x in re.findall('\d+', str(items[1]))]
            _file_size(dim1=dims[0], dim2=dims[1], dim3=dims[2], bits=8)
        if "b'    BoundingBox " in str(items):
            bounds = [float(x) for x in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(items[1]))]
    if len(bounds) == 0:
        print("Voxel resolution not found, setting to 1, 1, 1")
        spacing = (1, 1, 1)
    else:
        origin = bounds[::2]
        extent = bounds[1::2]
        physical_size = abs(np.array(origin)) + abs(np.array(extent))
        spacing = physical_size / np.array(dims)
        spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    return header, dims, spacing, origin, binary_length

def read_amira_data(amira_file, header_length=514):
    with open(amira_file, "rb") as binary_file:
        # Seek a specific position in the file and read N bytes
        binary_file.seek(header_length + 1)
        raw_binary = binary_file.read()
        if raw_binary[0:1] == b'\n':
            raw_binary = raw_binary[1:]
    return raw_binary

def write_amira_raw(amira_binary, out_name, out_dir=""):
    if "." in out_name:
        out_name.replace(".", "")
    if out_dir != "":
        out_name = pathlib.Path(out_dir).joinpath(out_name)
    with open(f"{out_name}.raw", "wb") as raw_data:
        raw_data.write(amira_binary)


def crude_threshold(inputImage, threshold=None):
    """
    Function to get a crude threshold based on the mean or median grey value of an image. If no value is passed, an
    estimate will be made using the mean and median of the intensity values.

    :param inputImage: A SimpleITK image.
    :param threshold: A value to threshold the image by.
    :return: Returns a thresholded SimpleITK image.
    """

    threshold = threshold

    nda = sitk.GetArrayFromImage(inputImage)

    if threshold == None:
        threshold_mean = nda.mean()
        threshold_median = np.median(nda)
        if threshold_mean > threshold_median:
            threshold = threshold_mean
        else:
            threshold = threshold_median
    crude_seg = inputImage > threshold
    return crude_seg

def thresh_simple(inputImage, background=0, foreground=1, outside=0, threads="threads"):
    start = timer()
    thresh = sitk.ThresholdImageFilter()
    thresh.SetNumberOfThreads(_get_threads(threads))
    thresh.SetLower(background)
    thresh.SetUpper(foreground)
    thresh.SetOutsideValue(outside)
    threshold = thresh.Execute(inputImage)
    print("\n")
    _end_timer(start, message="Simple threshold")
    return threshold

def _get_threads(threads):
    if threads == "threads":
        threads = (int(multiprocessing.cpu_count()) - 1)
    else:
        threads = int(threads)
    return threads

def check_label_values(inputFilename):
    inputImage = sitk.ReadImage(inputFilename)
    MinMax = sitk.MinimumMaximumImageFilter()
    MinMax.Execute(inputImage)
    pixelid = inputImage.GetPixelIDValue()
    if pixelid != 1:
        print(f"Rescaling to unsigned 8bit...")
        inputImage = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)

    imageMax = MinMax.GetMaximum()
    if imageMax != 255:
        print(f"Image max intensity is {imageMax}, rescaling to 255...\n")
        rescaleFilter = sitk.RescaleIntensityImageFilter()
        rescaleFilter.SetOutputMinimum(0)
        rescaleFilter.SetOutputMaximum(255)
        inputImage = rescaleFilter.Execute(inputImage)
    return inputImage



def check_if_label_exists(file_name, label_directory, verbose=True):
    image_file = file_name
    label_directory = pathlib.Path(label_directory)
    check_exists = pathlib.Path(image_file).parts[-1]
    check_file_type = check_exists.split(".")[0]
    #print(check_exists)
    if label_directory.joinpath(check_exists).exists():
        if verbose == True:
            print(f"File {check_exists} already exists...")
        return True
    elif label_directory.joinpath(f"{check_file_type}.tif").exists():
        if verbose == True:
            print(f"File {check_exists} already exists...")
        return True
    else:
        print(f"Standardizing {check_exists}")
        return False


def rescale_8(inputImage):
    """
    Takes in a SimpleITK image and rescales it to 8 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 8-bit SimpleITK formatted volume with gray values scaled between 0-255.
    """

    imageType = inputImage.GetPixelID()

    #Check to see if it is already unisgned 8 bit.
    if imageType == 1:
        print("Image is already unsigned 8...")
        scaled_8 = inputImage

    #If it isn't, go ahead and rescale.
    else:
        print("Rescaling to unsigned 8...")
        start = timer()
        scaled_8 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt8)
        _end_timer(start, message="Rescaling to unsigned 8")
    return scaled_8

def rgb_to_gray(image_array):
    gray_image = Image.fromarray(image_array)
    gray_image = ImageOps.grayscale(gray_image)
    gray_image_array = np.array(gray_image).astype(np.int8)
    return gray_image_array