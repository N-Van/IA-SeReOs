"""
Script to run the 3_class model in the command line. This should be executed using the pytorch_seg conda environment

Author: Sun, Yung-Chen yzs5463@psu.edu
Author: Yazdani, Amirsaeed auy200@psu.edu
Author: Nick Stephens nbs49@psu.edu

#If you get a win 95 error reinstall prompt-toolkit
python -m pip install -U prompt-toolkit~=2.0

"""
import os
import re
import sys
import glob
import math
import time
import torch
import socket
import pathlib
import platform
import numpy as np
import pandas as pd
import multiprocessing
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from PIL import UnidentifiedImageError
from timeit import default_timer as timer

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

# Provide the location of the net folder. This will work until packaged.
script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir))
#sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo\MARS\morphology\segmentation\pytorch_segmentation")
#sys.path.append(r"D:\Desktop\git_repo\MARS\morphology\segmentation\pytorch_segmentation")
from MARS.morphology.segmentation.pytorch_segmentation.net.unet_light_rdn import UNet_Light_RDN

class ReportPosition(sitk.Command):
    """
    class object to report progress in a consistent way.
    #Taken from https://simpleitk.readthedocs.io/en/master/link_FilterProgressReporting_docs.html
    """
    def __init__(self, po):
        # required
        super(ReportPosition, self).__init__()
        self.processObject = po

    def Execute(self):
        """
        :return: Prints a string with the progress to the console
        """
        print(f"\r           Progress:    {100 * self.processObject.GetProgress():03.1f}%", end='')

    def filterPosition(self, startorstop=""):
        """
        Function to print either start or stop within a filter.
        :param startorstop:
        :return:
        """
        if startorstop == 'start':
            print(f"\n{self.processObject.GetName()} executing....\nPlease stand by...")
        elif startorstop == 'stop':
            print("\nFiltering done!\n")
        else:
            print("Unknown command...")


def _setup_image(data_folder, image_name):
    """
    Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array
    image = Image.open(os.path.join(data_folder, image_name)).convert('L')
    image = np.array(image)

    # Check the dimensionality of the image, expand, transpose, for pytorch.
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
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
        print(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        print(f"{message} took: {float(elapsed):10.4f} seconds")

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

def _get_threads(threads):
    if threads == "threads":
        threads = (int(multiprocessing.cpu_count()) - 1)
    else:
        threads = int(threads)
    return threads

def _print_info(inputImage):
    """
    Function to return the basic information of an image volume.
    :param inputImage: A SimpleITK formated image volume.
    """
    image_type = inputImage.GetPixelIDTypeAsString()
    size = inputImage.GetSize()
    xdim, ydim, zdim = size[0], size[1], size[2]
    res = inputImage.GetSpacing()[0]
    if image_type == "8-bit unsigned integer":
        bits = 8
    elif image_type == "16-bit unsigned integer" or "16-bit signed integer":
        bits = 16
    elif image_type == "32-bit unsigned integer" or "32-bit signed integer":
        bits = 32
    else:
        bits = 64
    _file_size(xdim, ydim, zdim, bits)
    print(f"{image_type}\nx:{xdim} y:{ydim} z:{zdim}\nResolution:{res}\n")

def _setup_image(data_folder, image_name):
    """
    Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array
    image = Image.open(os.path.join(data_folder, image_name)).convert('L')
    image = np.array(image)

    # Check the dimensionality of the image, expand, transpose, for pytorch.
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    return image

def _save_predictors(pred, save_folder, image_name, file_type):
    """
    Internal function to convert predictions to an image and save in an output folder.
    """
    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    #File type dictionary for pillow
    type_dict = {"tif": "TIFF", "png": "PNG", "jpg": "JPEG", "bmp": "BMP"}
    f_type = type_dict[str(file_type)]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_img = pred_img.astype(np.uint8)
    pred_img = Image.fromarray(pred_img, 'L')
    pred_img.save(os.path.join(save_folder, f"{image_name[:-3]}.{str(file_type)}"), str(f_type))

def _setup_sitk_image(image_slice, direction="z"):
    """
    Internal function to read in an image and setup for classification by pytorch.
    """
    # Open the image using pillow and ensure it is grey scale ('L'), then turn it into a numpy array

    direction = str(direction).lower()

    #Convert the image slice into a numpy array
    image = sitk.GetArrayFromImage(image_slice)

    # Deal with the variation in the 3d versus 2d array.
    if len(image.shape) == 2:
        if direction == "z":
            #Expand the z axis
            image = np.expand_dims(image, axis=2)
            # Check the dimensionality of the image, expand, transpose, for pytorch.
            image = image.transpose((2, 0, 1))
        elif direction == "y":
            image = np.expand_dims(image, axis=1)
            image = image.transpose((1, 0, 2))
        else:
            image = np.expand_dims(image, axis=0)
            #image = image.transpose((0, 1, 2))
    return image

def _return_predictors(pred, direction="z"):
    """
    Internal function to convert predictions to an image and save in an output folder.
    """
    direction = str(direction).lower()

    # The dictionary for the grey value means for each class.
    # This will results in 0 for air, 128 for dirt, and 255 for bone.
    color_dict = [[0.0], [128.0], [255.0]]

    # Set up a blank numpy array to put the results into according to the values in the color_dict
    pred_img = np.zeros(pred.shape)
    for i in range(len(color_dict)):
        for j in range(len(color_dict[i])):
            pred_img[pred == i] = color_dict[i][0]

    # Cast the data as unsigned 8 bit and reconstruct the image for writing.
    pred_array = pred_img.astype(np.uint8)

    if direction == "z":
        pred_array = np.expand_dims(pred_array, axis=0)
    elif direction == "y":
        pred_array = np.expand_dims(pred_array, axis=1)
    else:
        pred_array = np.expand_dims(pred_array, axis=2)
    return pred_array


def _get_outDir(outDir):
    """
    Simple function to wrap an output directory using pathlib.
    :param outDir: Directory for writing out a file.
    :return:
    """
    if outDir == "":
        outDir = pathlib.Path.cwd()
    else:
        outDir = pathlib.Path(str(outDir))
    return outDir

def _get_inDir(inDir):
    """
    Simple function to wrap an input directory using pathlib.
    :param outDir: Directory for writing out a file.
    :return:
    """
    if inDir == "":
        inDir = pathlib.Path.cwd()
    else:
        inDir = pathlib.Path(str(inDir))
    return inDir

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

def read_image(inputImage):
    """
    Reads in various image file formats (mha, mhd, nia, nii, vtk, etc.) and places them into a SimpleITK volume format.
    :param inputImage: Either a volume (mhd, nii, vtk, etc.).
    :return: Returns a SimpleITK formatted image object.
    """
    print(f"Reading in {inputImage}.")
    start = timer()
    inputImage = sitk.ReadImage(str(inputImage))
    _end_timer(start, message="Reading in the image")
    _print_info(inputImage)
    print("\n")
    return inputImage

def write_image(inputImage, outName, outDir="", fileFormat="mhd"):
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
    outDir == _get_outDir(outDir)
    fileFormat = str(fileFormat)

    fileFormat = fileFormat.replace(".", "")
    outputImage = pathlib.Path(outDir).joinpath(str(outName) + "." + str(fileFormat))

    _print_info(inputImage)
    print(f"Writing {outName} to {outDir} as {fileFormat}.")
    sitk.WriteImage(inputImage, str(outputImage))
    _end_timer(start, message="Writing the image")

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
        image_slice = inputImage[:, slice,:]
    else:
        image_slice = inputImage[slice, :, :]
    return image_slice

def three_class_segmentation(inDir, outDir, outType, network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a pillow readable format (e.g. "tif", "png", "jpg", "bmp")
    :param inDir: The input directory where the images are located
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    # Check to make sure the output folder exists, and if it doesn't make it
    data_folder = inDir
    save_folder = outDir

    net = network

    # The file types that can be output along with the corresponding dictionary
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Get a list of files from the input folder using a list comprehension approach, then sort them numerically.
    image_names = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    image_names.sort()

    # Set up the count so there's something to watch as it processes
    seg_count = len(image_names)
    seg_now = seg_count
    print(f"Processing images in {data_folder}")

    # Loop through the images in the folder and use the image name for the output name
    for i in tqdm(range(len(image_names), unit=f" slices", desc=f" Segmenting...")):
        image_name = image_names[i]

        #Read the image in with pillow and set it as a numpy array for pytorch
        image = _setup_image(data_folder=data_folder, image_name=image_name)

        #Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        #Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()

        #Pass the predictions to be saved using pillow.
        _save_predictors(pred=pred, save_folder=outDir, image_name=image_name, file_type=outType)

        #Write out the information for the command line.
        seg_now -= 1
        per_complete = abs((1 - (seg_now / seg_count)) * 100)
        print(f'{seg_now} of {seg_count} remaining, {per_complete:3.2f}% complete...\r', end="")

    print('\n\nSegmentations are done!\n\n')

def three_class_segmentation_volume(inputImage, direction="z", network=""):
    """
    Function to segment a directory of 2d images using a pytorch model
    Images must be in a pillow readable format (e.g. "tif", "png", "jpg", "bmp")
    :param inDir: The input directory where the images are located
    :param outDir: The output directory. If this doesn't exist it will be created.
    :param outType: The output file type. Supported type are tif, png, jpg, and bmp.
    :return: Returns a segmented 2d image with grey values representing air, dirt, and bone.
    """
    net = network
    start = timer()
    # Set up the count so there's something to watch as it processes
    direction = str(direction).lower()

    if direction == "z":
        seg_count = inputImage.GetSize()[2]
    elif direction == "y":
        seg_count = inputImage.GetSize()[1]
    else:
        seg_count = inputImage.GetSize()[0]
    print(f"Processing {seg_count} slices...")

    # Create an empty volume to stuff the results into. A numpy approach was tested but proved to be slower
    vol_image = sitk.Image(inputImage.GetSize(), sitk.sitkUInt8)

    # Loop through the images in the folder and use the image name for the output name
    for i in tqdm(range(seg_count), unit=f" slices", desc=f" Segmenting {direction}"):
        image = feed_slice(inputImage, slice=i, direction=str(direction))

        #Read the image in with pillow and set it as a numpy array for pytorch
        image = _setup_sitk_image(image_slice=image, direction=direction)

        #Pass the numpy array to pytorch, convert to a float between 0-1,then copy into cuda memory for classifcation.
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float() / 255.0
        image = image.cuda()

        #Turn all the gradients to false and get the maximum predictors from the network
        with torch.no_grad():
            pred = net(image)
        pred = pred.argmax(1)
        pred = pred.cpu().squeeze().data.numpy()

        #Pass the predictions to be saved using pillow.
        pred = _return_predictors(pred=pred, direction=direction)
        slice_vol = sitk.GetImageFromArray(pred)
        #slice_vol = sitk.JoinSeries(slice)
        if direction == "z":
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, 0, i])
        elif direction == "y":
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[0, i, 0])
        else:
            vol_image = sitk.Paste(vol_image, slice_vol, slice_vol.GetSize(), destinationIndex=[i, 0, 0])


    #vol_image = empty_slice[1:]
    print('\n\nSegmentations are done!\n\n')
    _end_timer(start_timer=start, message="Segmentations")
    return vol_image

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
        print("Image is already unsigned 16...")
        scaled_16 = inputImage
    else:
        # Read in the other image and recast to float 32
        print("Rescaling to unsigned 16...")
        start = timer()
        scaled_16 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkUInt16)
        _print_info(scaled_16)
        _end_timer(start, message="Rescaling to unsigned 16")
    return scaled_16

def rescale_32(inputImage):
    """
    Takes in a SimpleITK image and rescales it to 32 bit.
    :param inputImage: A SimpleITK formatted volume.
    :return: Returns an unsigned 16-bit SimpleITK formatted volume with 2^64 distinct gray values.
    """
    imageType = inputImage.GetPixelID()
    if imageType == 8:
        print("Image is already float 32...")
        scaled_32 = inputImage
    else:
        # Read in the other image and recast to float 32
        print('Rescaling to float 32...')
        start = timer()
        scaled_32 = sitk.Cast(sitk.RescaleIntensity(inputImage), sitk.sitkFloat32)
        _print_info(scaled_32)
        _end_timer(start, message="Rescaling to 32-bit float")
    return scaled_32

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

def three_class_seg_xyz(inputImage, network=""):
    #Segment the volume from all three directions
    seg_z = three_class_segmentation_volume(inputImage=inputImage, direction="z", network=network)
    seg_y = three_class_segmentation_volume(inputImage=inputImage, direction="y", network=network)
    seg_x = three_class_segmentation_volume(inputImage=inputImage, direction="x", network=network)

    #Rescale them to prevent overflow when we combine
    seg_z = rescale_16(seg_z)
    seg_y = rescale_16(seg_y)
    seg_z = combine_images(seg_z, seg_y)

    # Free up memory
    seg_y = 0
    seg_x = rescale_16(seg_x)

    seg_z = combine_images(seg_z, seg_x)

    seg_x = 0
    #Get the final product
    seg = rescale_8(seg_z)
    return seg

def _set_filter_events(sitkfilter):
    """
    Internal function to setup the filter events passed to it. Uses the ReportPosition class.
    :param sitkfilter: A simpleITK filter
    :return: Returns a filter loaded with event commands.
    """
    filter_events = ReportPosition(sitkfilter)
    sitkfilter.AddCommand(sitk.sitkStartEvent, filter_events)
    sitkfilter.AddCommand(sitk.sitkProgressEvent, filter_events)
    sitkfilter.AddCommand(sitk.sitkProgressEvent, lambda: sys.stdout.flush())
    return sitkfilter, filter_events


def thresh_simple(inputImage, background=0, foreground=1, outside=0, threads="threads"):
    start = timer()
    thresh = sitk.ThresholdImageFilter()
    thresh.SetNumberOfThreads(_get_threads(threads))
    thresh.SetLower(background)
    thresh.SetUpper(foreground)
    thresh.SetOutsideValue(outside)
    thresh, filter_events = _set_filter_events(thresh)
    threshold = thresh.Execute(inputImage)
    print("\n")
    _end_timer(start, message="Simple threshold")
    return threshold

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

def read_stack(inputStack):
    """
    Reads in a series of images and then places them into a SimpleITK volume format.

    :param inputStack: A stack of images (e.g. tif, png, etc).
    :return: Returns a SimpleITK formatted image object.
    """
    # Read in the other image and recast to float 32
    start = timer()
    print("Reading in files...")
    inputStack.sort()
    inputStack = sitk.ReadImage(inputStack)
    _end_timer(start, message="Reading in the stack")
    _print_info(inputStack)
    print("\n")
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
    print(f"Reading in {len(inputStack)} DICOM images...")

    inputStack.sort()
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(inputStack)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    sitk_image = series_reader.Execute()

    _print_info(sitk_image)

    #Grab the metadata, Name, ID, DOB, etc.
    direction = sitk_image.GetDirection()
    tags_to_copy = ["0010|0010", "0010|0020", "0010|0030", "0020|000D", "0020|0010",
                    "0008|0020", "0008|0030", "0008|0050", "0008|0060"]
    process_tag = ["0008|103e"]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    series_tag_values = [(k, series_reader.GetMetaData(0, k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0, k)]

    modified_tags = [("0008|0031", modification_time), ("0008|0021", modification_date), ("0008|0008", "DERIVED\\SECONDARY"),
                     ("0020|000e", "" + modification_date + ".1" + modification_time),
                     ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                       direction[1], direction[4], direction[7]))))]

    series_tag_values = series_tag_values + modified_tags

    #Inset the new processing data
    if series_reader.HasMetaDataKey(0, process_tag[0]) == True:
        series_tag_values = series_tag_values + [("0008|103e", series_reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]
    else:
        series_tag_values = series_tag_values + [("0008|103e", "Processed-SimpleITK")]

    #To prevent the stacking of the same processing information
    if series_tag_values[-1] == ('0008|103e', 'Processed-SimpleITK  Processed-SimpleITK'):
        series_tag_values[-1] = ("0008|103e", "Processed-SimpleITK")
    _end_timer(start_timer=start, message="Reading DICOM stack")
    return sitk_image, series_tag_values

def write_dicom(inputImage, metadata, outName, outDir=""):
    """
    """
    #Modified from: https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

    start = timer()
    series_tag_values = metadata
    if outDir == "":
        outDir = pathlib.Path.cwd()
    else:
        outDir = pathlib.Path(outDir)

    #Make is so the file name generator deal with these parts of the name
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

    for i in tqdm(range(slice_num), unit=" slices", desc=f" Writing out {slice_num} DICOM slices to {outDir}..."):
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
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, inputImage.TransformIndexToPhysicalPoint((0, 0, i)))))
        #   Instance Number
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        writer.SetFileName(f'{str(outName)}_{i:0{int(digits_offset)}}.dcm')
        writer.Execute(image_slice)
    print("\n")
    _end_timer(start, message="Writing DICOM slices")

