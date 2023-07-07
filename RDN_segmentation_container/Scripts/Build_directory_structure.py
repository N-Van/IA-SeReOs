"""
Python 3.6+ Script to simplify the process of generating a par file from an existing data structure. Uses glob to
recursively find a specific file string in a base folder. It then splits off the file type and the location.
Thereafter it will attempt to find the bone, portion, and side. If it doesn't work, there may be some manual
fiddling involved. When formatted properly, you can generate a par file from the inventory.

#To turn off the annoying warning
pd.options.mode.chained_assignment = None  # default='warn'

Author: Nick Stephens (nbs49@psu.edu)
Author: Lily DeMars (lvd5263@psu.edu)
"""


import os
import re
import sys
import glob
import shutil
import pathlib
import platform
import numpy as np
import pandas as pd
from time import time as timer

########################################
#                                      #
# Functions to be pasted to a console #
#                                      #
########################################
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


def get_initial_dataframe(matching_key, directory=""):
    """
    Function to return a pandas dataframe from a file structure using pathlibs recursive glob
    :param matching_key:
    :return: Returns a pandas dataframe with the location, file, and file_type
    :usage: df = get_intial_dataframe("*0001.tif")
    """
    start = timer()
    # Create a dictionary to append to with the key Location
    if directory == "":
        directory = pathlib.Path.cwd()
        print(
            f"\nNo directory provided, using the current working directory:\n {directory}\n"
        )
    else:
        print(f"Searcing in {directory}...")

    files = {"Location": []}
    matching_key = str(matching_key)
    # Use the rglob module within pathlib to recursively find all the files matching a wildcarded key word
    # This is the preferred method, but Long file names in Windows don't work.
    # for filename in pathlib.Path(directory).rglob(str(matching_key)):
    # #Append the found tiles to the dictionary
    # files["Location"] += [str(filename)]

    print(
        f"\nSearching for {matching_key}, which can take some time...\n please standby..."
    )
    for filename in glob.glob("**/*" + str(matching_key), recursive=True):
        files["Location"] += [str(filename)]
        print(filename)
    # Convert the dictionary to a pandas dataframe
    files = pd.DataFrame(files)

    if len(files) == 0:
        print("No matching {} files found...".format(str(matching_key)))
    else:
        files["Root"] = str(directory)
        # Take the column and reverse split by the backslash, which you need two of to re-enter the expression.
        # You can control how many splits are done with the n= expression
        if platform.system().lower() == "windows":
            files["Location"] = np_concat(
                files["Root"], files["Location"], join_char="\\"
            )
            files = files["Location"].str.rsplit("\\", expand=True, n=1)
        else:
            files["Location"] = np_concat(
                files["Root"], files["Location"], join_char="/"
            )
            files = files["Location"].str.rsplit("/", expand=True, n=1)

        # Split the file_name column by reverse to isolate the tif and the file name proper
        files2 = files[1].str.rsplit(".", expand=True, n=1)

        # Place the two rows into the original dataframes columns
        files[["File_name", "File_type"]] = files2[[0, 1]]
        files.drop(1, axis=1, inplace=True)

        files.columns = ["Location", "File_name", "File_type"]
        _end_timer(start, message="Searching for files")
        return files


def build_from_multiple(file_type_list, directory=""):
    """
    If you have multiple file types to match you can loop through a list to build the initial dataframe.
    :param file_type_list: A list of file types to text match (e.g. ["0001.tif", "0001.dcm", ".vol", ".raw"])
    :return: Returns a dataframe comprised of matching file types.
    """

    df = pd.DataFrame()

    if directory == "":
        directory = pathlib.Path.cwd()
        print(
            f"\nNo directory provided, using the current working directory:\n {directory}\n"
        )
    else:
        directory = pathlib.Path(directory)

    for f in file_type_list:
        print(f"\n\nSearching for {f} files....")
        df_found = get_initial_dataframe(str("*" + str(f)), directory=directory)
        df = pd.concat([df, df_found])
    df.reset_index(drop=True, inplace=True)
    return df


def setup_individual_nf(pandas_df, match_location="temp"):
    df = pandas_df
    match_location = str(match_location)
    individual_replace = {
        "819932": "20",
        "819941": "27",
        "819951": "33",
        "819957": "37",
        "819963": "41",
        "819977": "44",
        "819983": "45",
        "819994": "49",
        "819996": "50",
        "820647": "63",
        "820652": "66",
        "820658": "69",
        "820668": "71",
        "820696": "80",
        "820715": "90",
        "820735": "105",
        "820740": "106",
        "821042": "132",
        "821228": "216",
        "821230": "217",
        "819920": "14",
        "820655": "68",
        "821126": "185",
        "821217": "210",
        "821290": "235",
    }
    df["Individual"] = df[match_location].map(individual_replace)
    return df


def setup_bone_columns(pandas_df, match_location="File_name", shafts=False):
    """
    Function do some initial categorization of bones based on the files names. Be careful, because people have
    many many many different ways of coding bones and anatomical regions.
    :param dataframe: A pandas dataframe with a ["File_name"] column
    :return: Returns a dataframe with the bone, portion, and side if they"re present.
    :useage: df = setup_bone_columns(df)
    """

    df = pandas_df.copy()
    # Use a pattern for the various bone names and portions people use
    bone_pattern = "fem|femur|fumer|_df_|_pf_|_dh_|_ph_|tib|tibia|c7|c6|vert|hum|humerus|calc|talus|patella"
    portion_pattern = "dist|distal|prox|proximal|humerusp|humerud|femurp|femurd|_df_|_pf_|_dh_|_ph_|_hd_|tibiap|tibiad|body|whole|head|base|medial|lateral|calc|talus|c6|c7|t1|vert|diaphysis|shaft|merged|mid"
    side_pattern = " l | r | l_| r_|_l_|_r_|left|right|humerusp_r|humerusd_r|humerusd_r|hum_prox_r|hum_dist_r|hum_prox_l|hum_dist_l|humerusd_l|femurp_r|femurp_l|femurd_r|femurd_l|tibiap_r|tibiap_l|tibiad_r|tibiad_l|lhumerusdist|ltibiadist|Lfemurdist|lhumerusprox|ltibiaprox|lfemurprox|lfemurdist|rhumerusdist|rtibiadist|rfemurprox|rfemurdist|rhumerusprox|rtibiaprox|rfemurprox|rfemurmid|lfemurmid|rhumerusmid|lhumerusmid|rtibiamid|ltibiamid"

    if match_location != "File_name":
        match_location = str(match_location)
        df["Bone"] = (
            df[str(match_location)]
            .str.lower()
            .str.extract("(" + bone_pattern + ")", expand=False)
        )
        df["Portion"] = (
            df[str(match_location)]
            .str.lower()
            .str.extract("(" + portion_pattern + ")", expand=False)
        )
        df["Side"] = (
            df[str(match_location)]
            .str.lower()
            .str.extract("(" + side_pattern + ")", expand=False)
        )
    else:
        df["Bone"] = (
            df["File_name"]
            .str.lower()
            .str.extract("(" + bone_pattern + ")", expand=False)
        )
        df["Portion"] = (
            df["File_name"]
            .str.lower()
            .str.extract("(" + portion_pattern + ")", expand=False)
        )
        df["Side"] = (
            df["File_name"]
            .str.lower()
            .str.extract("(" + side_pattern + ")", expand=False)
        )

    # Replaces those found values using a mapped dictionary. The idea is to make things uniform.
    side_replace = {
        " l ": "L",
        "left": "L",
        " l_": "L",
        "_l_": "L",
        "humerusp_l": "L",
        "femurp_l": "L",
        "tibiap_l": "L",
        "humerusd_l": "L",
        "femurd_l": "L",
        "tibiad_l": "L",
        "lhumerus": "L",
        "ltibia": "L",
        "lfemur": "L",
        "hum_dist_l": "L",
        "hum_prox_l": "L",
        "lhumerusdist": "L",
        "ltibiadist": "L",
        "lfemurdist": "L",
        "lhumerusprox": "L",
        "ltibiaprox": "L",
        "lfemurprox": "L",
        "lhumerusmid": "L",
        "ltibiamid": "L",
        "lfemurmid": "L",
        " r ": "R",
        "right": "R",
        " r_": "R",
        "_r_": "R",
        "humerusp_r": "R",
        "femurp_r": "R",
        "tibiap_r": "R",
        "humerusd_r": "R",
        "femurd_r": "R",
        "tibiad_r": "R",
        "hum_dist_r": "R",
        "hum_prox_r": "R",
        "rhumerus": "R",
        "rtibia": "R",
        "rfemur": "R",
        "rhumerusdist": "R",
        "rtibiadist": "R",
        "rfemurdist": "R",
        "rhumerusprox": "R",
        "rtibiaprox": "R",
        "rfemurprox": "R",
        "rhumerusmid": "R",
        "rtibiamid": "R",
        "rfemurmid": "R",
    }

    df["Side"] = df["Side"].map(side_replace)

    bone_replace = {
        "t1": "T1",
        "c6": "C6",
        "c7": "C7",
        "vert": "Vert",
        "C7": "C7",
        "tib": "Tibia",
        "rtibia": "Tibia",
        "ltibia": "Tibia",
        "ltibia": "Tibia",
        "rtibia": "Tibia",
        "fem": "Femur",
        "fumer": "Femur",
        "_df_": "Femur",
        "_pf_": "Femur",
        "lfemur": "Femur",
        "rfemur": "Femur",
        "hum": "Humerus",
        "_dh_": "Humerus",
        "_ph_": "Humerus",
        "lhumerus": "Humerus",
        "rhumerus": "Humerus",
        "calc": "Calc",
        "talus": "Talus",
        "patella": "Patella",
    }
    df["Bone"] = df["Bone"].map(bone_replace)

    if shafts == True:
        portion_replace = {
            "body": "Whole",
            "whole": "Whole",
            "head": "ShaftProx",
            "prox": "ShaftProx",
            "humerusp": "ShaftProx",
            "femurp": "ShaftProx",
            "_pf_": "ShaftProx",
            "_hd_": "ShaftProx",
            "_ph_": "ShaftProx",
            "tibiap": "ShaftProx",
            "mid": "ShaftMid",
            "rfemurmid": "ShaftMid",
            "lfemurmid": "ShaftMid",
            "ltibiamid": "ShaftMid",
            "rtibiamid": "ShaftMid",
            "ltibiamid": "ShaftMid",
            "dist": "ShaftDist",
            "base": "ShaftDist",
            "humerud": "ShaftDist",
            "femurd": "ShaftDist",
            "tibiad": "ShaftDist",
            "_dh_": "ShaftDist",
            "_df_": "ShaftDist",
            "med": "Medial",
            "lat": "Lateral",
            "talus": "Whole",
            "calc": "Whole",
            "c6": "Whole",
            "c7": "Whole",
            "t1": "Whole",
            "vert": "Whole",
            "shaft": "Overview",
            "diaphysis": "Overview",
            "merged": "Overview",
        }
        df["Portion"] = df["Portion"].map(portion_replace)
    else:
        portion_replace = {
            "body": "Whole",
            "whole": "Whole",
            "head": "Prox",
            "prox": "Prox",
            "humerusp": "Prox",
            "femurp": "Prox",
            "_pf_": "Prox",
            "_hd_": "Prox",
            "_ph_": "Prox",
            "tibiap": "Prox",
            "dist": "Dist",
            "base": "Dist",
            "humerud": "Dist",
            "femurd": "Dist",
            "tibiad": "Dist",
            "_dh_": "Dist",
            "_df_": "Dist",
            "med": "Medial",
            "lat": "Lateral",
            "talus": "Whole",
            "calc": "Whole",
            "c6": "Whole",
            "c7": "Whole",
            "t1": "Whole",
            "vert": "Whole",
            "shaft": "Overview",
            "diaphysis": "Overview",
            "merged": "Overview",
        }
        df["Portion"] = df["Portion"].map(portion_replace)

    # Sort the values
    df.sort_values(by=["Bone", "Portion", "Side"], inplace=True)

    # Reset the index so they're sequential again
    df.reset_index(drop=True, inplace=True)

    return df


def remove_unwanted_in_column(pandas_df, df_column="", match_list=""):
    """
    Fucntion to remove rows in a column using string matching. Accepts a list of case sensitive
    strings and a column name in a pandas dataframe. If a list isn't provided, this will default
    to a column named 'Location", and use the list  ["VOI", "IMJ", "CentreSlice", "Sinograms"].

    :param pandas_df:
    :param df_column:
    :param match_list:
    :return:
    """
    df = pandas_df.copy()
    print(f"\n\nInitial dataframe had {df} rows.\n")
    before = len(df)
    if df_column == "":
        df_column = "Location"
    else:
        df_column = str(df_column)
    if match_list == "":
        match_list = ["VOI", "IMJ", "CentreSlice", "Sinograms"]
    else:
        match_list = list(match_list)

    # Drop rows that don't contain (i.e. ~ ) a string.
    remove_pattern = "|".join(match_list)
    for f in remove_pattern.split("|"):
        print(f"Removing rows with the string matching {f} from df['Location']....\n")
    df = df[
        ~df[str(df_column)].str.contains(
            "(" + remove_pattern + ")", case=True, regex=True
        )
    ]
    after = len(df)
    difference = int(before - after)
    print(
        f"\nThe new dataframe has {after} rows.\n                      {difference} removed.\n\n"
    )
    df.reset_index(drop=True, inplace=True)
    return df


# first define a function: given a Series of string, split each element into a new series
def split_series(dataframe_series, text_seperator):
    """
    Splits a dataframe series and then uses grouby to rebuild it to include all the other information.
    credit eyllansec: https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows
    :param dataframe_series: series from a pandas dataframe
    :param text_seperator: Portion of the string to split into elements.
    :return:
    :usage:
    df2=(df.groupby(df.columns.drop("Individual").tolist()) #group by all but one column
          ["Individual"] #select the column to be split
          .apply(split_series,sep="_") # split "Seatblocks" in each group
         .reset_index(drop=True,level=-1).reset_index()) #remove extra index created
    """
    return pd.Series(
        dataframe_series.str.cat(sep=text_seperator).split(sep=text_seperator)
    )


def explode(pandas_df, cols, split_on=","):
    """
    Explode dataframe on the given column, split on given delimeter.
    credit (titipata and piRSquared https://stackoverflow.com/questions/38651008/splitting-multiple-columns-into-rows-in-pandas-dataframe)
    :param df: pandas dataframe
    :param cols: columns you would like to split
    :param split_on: text character you would liek to split
    :return: Returns a split dataframe with the columns copied over
    :usage:
    new_df = explode(df, ["value", "date"])
    """
    df = pandas_df.copy()
    cols_sep = list(set(df.columns) - set(cols))
    df_cols = df[cols_sep]
    explode_len = df[cols[0]].str.split(split_on).map(len)
    repeat_list = []
    # for r, e in zip(df_cols.as_matrix(), explode_len):
    #    repeat_list.extend([list(r)]*e)
    for r, e in zip(df_cols.values, explode_len):
        repeat_list.extend([list(r)] * e)
    df_repeat = pd.DataFrame(repeat_list, columns=cols_sep)
    df_explode = pd.concat(
        [
            df[col]
            .str.split(split_on, expand=True)
            .stack()
            .str.strip()
            .reset_index(drop=True)
            for col in cols
        ],
        axis=1,
    )
    df_explode.columns = cols
    return pd.concat((df_repeat, df_explode), axis=1)


def numpy_concat(*args, join_char="_"):
    """
    Function to deal with nan values when concatenating portions of a dataframe.
    :param args:
    :return: Returns an array that can be placed in a dataframe.
    """
    strs = [str(arg) for arg in args if not pd.isnull(arg)]
    return str(join_char).join(strs) if strs else np.nan


def get_par(pandas_df, root_folder="Z:/RyanLab/Projects/nsf_human_variation/"):
    """
    Function to generate an initial par file for Medtool with default values.
    :param pandas_df: A pandas dataframe containing the columns needed to generate the correct values.
    :param root_folder: The folder where the project is based. Default set to "Z:/RyanLab/Projects/nsf_human_variation/".
    :return: Returns a dataframe that is ready to be loaded into medtool.
    usage: Requires the columns ["Population"], ["Individual"], ["Bone"], ["Portion"], ["Side"], ["Species"]
    ["File_name"], ["File_type"]
    """
    df = pandas_df.copy()
    # Create the oldname column from the relevant columns and set values where needed
    df["$oldname"] = np_concat(
        df["Population"],
        df["Individual"],
        df["Burial_Part_Accession"],
        df["Bone"],
        df["Portion"],
        df["Side"],
    )
    df["$name"] = np_concat(
        df["Population"], df["Individual"], df["Bone"], df["Side"], join_char=""
    )
    df["$res"] = float(0)
    df["$dim1"] = int(0)
    df["$dim2"] = int(0)
    df["$dim3"] = int(0)
    df["$kc"] = df["KC"].astype(int)
    df["$kpoint"] = int(0)
    df["$kout"] = int(3)
    df["$kin"] = int(6)
    df["$grid"] = int(200)
    df["$kmeans"] = int(2)
    df["$miathresh"] = int(1)
    df["$probability"] = float(0.02)
    df["$threads"] = float(2)
    df["$inmesh"] = float(0.6)
    df["$outmesh"] = float(0.3)
    df["$pcinmehs"] = df["pcinmesh"]
    df["$z1"] = int(0)
    df["$z2"] = int(0)
    df["$cut"] = int(0)
    df["$r1"] = int(0)
    df["$r2"] = int(0)
    df["$r3"] = int(0)
    df["$r3"] = int(0)
    df["$r4"] = int(0)
    df["$r5"] = int(0)
    df["$r6"] = int(0)
    df["$r7"] = int(0)
    df["$r7"] = int(0)
    df["$r8"] = int(0)
    df["$r9"] = int(0)
    df["$species"] = df["Species"]
    df["$population"] = df["Population"]
    df["$specimen"] = df["Individual"]
    df["$bone"] = df["Bone"]
    df["$portion"] = df["Portion"]
    df["$path"] = np_concat(
        str(root_folder),
        df["Species"],
        df["Population"],
        df["Individual"],
        df["Bone"],
        df["$portion"],
        join_char="/",
    )
    df["$path"] = df["$path"].str.replace(re.escape("//"), "/")
    df["$folder"] = df["$path"].str.replace(re.escape("/"), re.escape("\\"))
    df["$segmentation"] = df["$path"].str.replace("Z:/", "/mnt/ics/")
    df["$location"] = df["Location"]
    df["$file"] = df["File_name"]
    df["$type"] = df["File_type"]
    df["$thresh"] = int(0)
    bits_pattern = "tif|Tif|dcm|dicom|Dicom|vol|Vol|raw|Raw|nii"
    df["$bits"] = (
        df["$type"].str.lower().str.extract("(" + bits_pattern + ")", expand=False)
    )
    # Replaces those found values using a mapped dictionary with something consistent
    bits_replace = {
        "tif": "MET_USHORT",
        "dicom": "MET_CHAR",
        "vol": "MET_FLOAT",
        "raw": "MET_USHORT",
        "nii": "MET_USHORT",
    }
    df["$bits"] = df["$bits"].map(bits_replace)
    df["$oldname"] = df["$oldname"].str.replace("__", "_")
    # Use lambda to take from the last character over if it matches "_" otherwise take the whole thing
    df["$oldname"] = df["$oldname"].apply(lambda x: x[:-1] if x[-1:] == "_" else x)
    df["$folder"] = df["$folder"].str.replace("__", "")
    df["$path"] = df["$path"].str.replace("__", "")
    df["$segmentation"] = df["$path"].str.replace("__", "")
    # df.drop(columns="$oldname2")
    df = df.loc[:, df.columns.str.startswith("$")]
    return df


def linuxify_par(par_file, base_folder, linux_base_folder):
    """
    Convert a MedTool parameter file in windows format to linux format. Predominantly is concerned with fixing file paths.
    :param par_file: a MedTool formatted parameter file read in as a pandas data frame with the column [$folder]
    :param base_folder: The initial folder you want to replace.
    :param linux_base_folder: The folder naming convention you want to replace the base folder with.
    :return: Returns a MedTool parameter file with linux style file structure.
    :useage:
            #Convert the windows par to linux
            base_folder = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation")
            linux_base_folder = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/")

            #Turn it into a linux par and write it out.
            linux_par = linuxify_par(par, base_folder, linux_base_folder)
            linux_par.to_csv(population + "_linux.par", index=False, sep=";")

    """
    base_folder = pathlib.Path(base_folder).as_posix()
    linux_base_folder = pathlib.Path(linux_base_folder).as_posix()

    # Ensure these are strings and replace the backslashes with forward slashes
    par_file["$folder"] = (
        par_file["$folder"].astype(str).replace(r"\\", "/", regex=True)
    )
    par_file["$path"] = par_file["$path"].astype(str).replace(r"\\", "/", regex=True)
    par_file["$segmentation"] = (
        par_file["$segmentation"].astype(str).replace(r"\\", "/", regex=True)
    )
    par_file["$location"] = (
        par_file["$location"].astype(str).replace(r"\\", "/", regex=True)
    )

    # Use a lamda function to map a pathlib object onto the column of the dataframe
    folder_replace = lambda x: pathlib.Path(
        str(x).replace(str(base_folder), str(linux_base_folder))
    ).as_posix()
    par_file = par_file.applymap(folder_replace)

    return par_file


def print_df(pandas_df):
    """
    Function to quickly print all rows of a dataframe.
    :param pandas_df:pandas dataframe or subset of a dataframe df["Bone], or df[["Individual", "Bone"]], etc.
    :return: Prints the complete dataframe to the console
    """
    df = pandas_df.copy()
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(df)


def clean_temp(pandas_series):
    """
    Function to clean up a pandas series prior to splitting.
    :param pandas_series:
    :return:
    """
    df["temp"] = pandas_series
    df["temp"] = df["temp"].str.replace(", ", "_")
    df["temp"] = df["temp"].str.replace(".", "_")
    df["temp"] = df["temp"].str.replace(" - ", "_")
    df["temp"] = df["temp"].str.replace("-", "_")
    df["temp"] = df["temp"].str.replace("---", "_")
    df["temp"] = df["temp"].str.replace(" ", "_")
    df["temp"] = df["temp"].str.replace("___", "_")
    df["temp"] = df["temp"].str.replace("__", "_")
    return df["temp"]


def isolate_shafts(pandas_df, search_column="Location"):
    """
    Function to isolate just the shafts in a datafame. Uses the "location" column by default. Otherwise pass the column
    header you wish to use (e.g. "File_name")
    :param pandas_df:
    :return:
    """
    df = pandas_df.copy()
    if search_column != "Location":
        search_column = str(search_column)
    else:
        search_column = "Location"
    string_match = "diaphysis|Diaphysis|shaft|Shaft"
    df2 = df[df[str(search_column)].str.contains(string_match)]
    return df2


def remove_projections(pandas_df, fldr_substring=["_01", "_02", "_03"]):
    """
    Function to remove the projection folders and retain only the nested recontructed folders from a dataframe.
    :param pandas_df: pandas dataframs
    :param fldr_substring: A list of strings to match
    :return:
    """

    df = pandas_df.copy()

    if len(fldr_substring) != 3:
        fldr_substring = list(fldr_substring)
        print("Isolating folders matching {}...".format(fldr_substring))
    else:
        fldr_substring = list(fldr_substring)
        print("Isolating folders matching {}...".format(fldr_substring))

    pattern_match = "(" + "|".join(fldr_substring) + ")"

    df = df[df["Location"].str.contains(pattern_match)]
    return df


def setup_temp(pandas_df):
    """
    Funciton to quickly replace some common abbreviations.
    :param pandas_df:
    :return:
    """
    df = pandas_df.copy()
    df["temp"] = df["temp"].str.lower()
    df["temp"] = clean_temp(df["temp"])
    df["temp"] = df["temp"].str.replace("tib_", "tibia_")
    df["temp"] = df["temp"].str.replace("fem_", "femur_")
    df["temp"] = df["temp"].str.replace("hum_", "humerus_")
    return df


def set_print_size_max():
    columns, rows = shutil.get_terminal_size()
    columns = int(columns) - 6
    pd.options.display.max_colwidth = int(columns)
    print("Pandas print size width set to {}".format(columns))


def pre_explode_stacked(pandas_df):
    """
    This is a lazy and messy way to quickly clean up a temp column so it is a format that explode will use.
    :param pandas_df: Pandas dataframe.
    :return:
    """

    df = pandas_df.copy()
    df["temp"] = df["temp"].str.lower()
    df["temp"] = df["temp"].str.replace(
        "rhumerus_prox_lhumerus_dist_shaft", "rhumerusprox_lhumerusdist"
    )
    df["temp"] = df["temp"].str.replace(
        "rhumerus_dist_lhumerus_prox", "rhumerusdist_lhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "lhumerus_dist_ltibia_dist", "rhumerusprox_lhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "lhumerus_prox_ltibia_prox", "lhumerusprox_ltibiaprox"
    )
    df["temp"] = df["temp"].str.replace(
        "lhumerus_dist_ltibia_dist", "lhumerusdist_ltibiadist"
    )
    df["temp"] = df["temp"].str.replace(
        "rhumerus_dist_lhumerus_prox", "rhumerusdist_lhumerusdist"
    )
    df["temp"] = df["temp"].str.replace(
        "humerus_prox_l_dist_r", "lhumerusprox_rhumerusdist"
    )
    df["temp"] = df["temp"].str.replace(
        "humerus_prox_r_dist_l", "rhumerusdist_lhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "humerus_dist_l_prox_r", "lhumerusdist_rhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "humerus_dist_r_prox_l", "rhumerusdist_lhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "tibia_dist_humerus_prox_l", "tibiadist_lhumerusprox"
    )
    df["temp"] = df["temp"].str.replace(
        "tibia_prox_humerus_dist_l", "tibiaprox_lhumerusdist"
    )
    df["temp"] = df["temp"].str.replace("humerus_dist_r", "rhumerusdist")
    df["temp"] = df["temp"].str.replace("humerus_dist_l", "lhumerusdist")
    df["temp"] = df["temp"].str.replace("humerus_prox_r", "rhumerusprox")
    df["temp"] = df["temp"].str.replace("humerus_prox_l", "lhumerusprox")
    df["temp"] = df["temp"].str.replace("rfemur_mid", "rfemurmid")
    df["temp"] = df["temp"].str.replace("lfemur_mid", "lfemurmid")
    df["temp"] = df["temp"].str.replace("femur_mid", "femurmid")
    df["temp"] = df["temp"].str.replace("femur_dist", "femurdist")
    df["temp"] = df["temp"].str.replace("femur_prox", "femurprox")
    df["temp"] = df["temp"].str.replace("humerus_l", "lhumerus")
    df["temp"] = df["temp"].str.replace("humerus_r", "rhumerus")
    df["temp"] = df["temp"].str.replace("femur_l", "lfemur")
    df["temp"] = df["temp"].str.replace("femur_r", "rfemur")
    df["temp"] = df["temp"].str.replace("tibia_r", "rtibia")
    df["temp"] = df["temp"].str.replace("tibia_l", "ltibia")
    df["temp"] = df["temp"].str.replace("_dist", "dist")
    df["temp"] = df["temp"].str.replace("_prox", "prox")
    df["temp"] = df["temp"].str.replace("_mid", "mid")
    # df["temp"] = df["temp"].str.replace("", "")
    # df["temp"] = df["temp"].str.replace("", "")
    # df["temp"] = df["temp"].str.replace("", "")
    return df


def reassign_cell(pandas_df, column_name, location_list, assingment):
    """
    Assigns a string to a cell by the index number.
    :param pandas_df:
    :param column_name:
    :param location_list:
    :param assingment:
    :return:
    """
    df = pandas_df
    for f in location_list:
        df[str(column_name)].loc[int(f)] = str(assingment)


# Run to vectorize the numpy concat funciton
np_concat = np.vectorize(numpy_concat)

########################################
#                                      #
# This is where we actually do stuff   #
#                                      #
########################################

# Set pandas to print long strings
set_print_size_max()

# To turn off the annoying warning
pd.options.mode.chained_assignment = None  # default='warn'

# Get the directory where the files are that you want to work with
directory = pathlib.Path(r"Z:\RyanLab\Projects\NStephens\SSRI CT Scans")
# directory = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Data/USC transfer/Skull_Gultch")

# Change to the directory and print it out.
os.chdir(directory)
print(pathlib.Path.cwd())

# Enter in the file type to be searched (e.g. tif, vol, mhd, nii, raw... etc), use if you know all of the scans you are interested in have the same file type
file_type = ".bmp"

# Define the species
species = "Mus_musculus"

# Define what population you are working on
population = "SSRI"

# Establish the initial structure
df = get_initial_dataframe("*" + file_type)

# Then try to grab the bone, side, and portion information
df = setup_bone_columns(df)

# Force everything to be lowercase so it's easier for the remainder
df["temp"] = df["File_name"].str.lower()


########################################
#                                      #
# If you have multiple file types      #
#                                      #
########################################

# Define the file types to search for
file_type_list = ["0001.tif", "0001.dcm", ".vol", ".raw"]

# Build a dataframe from the matches
df = build_from_multiple(file_type_list=file_type_list)

# Set up common phrases for reconstructions that we aren't interested in
match_list = ["VOI", "IMJ", "CentreSlice", "Sinograms"]

# Then remove those
df = remove_unwanted_in_column(df, df_column="Location", match_list=match_list)

# If the reconstructions are nested (i.e. with a Nikon scanner) isolate just those folders.
# df = remove_projections(df, fldr_substring=["_01", "_02", "_03"])
# df.reset_index(drop=True, inplace=True)

# Assign bones, side, and portion by common match phrases.
df = setup_bone_columns(df)

# To see an entire column
print_df(df[["File_name", "Bone", "Portion", "Side"]])

df_tif = df[(df.File_type == "tif")]

# If there are projections, get rid of them
df_tif = remove_projections(df_tif)

# Subset all the things that are not tifs
df_other = df[~(df.File_type == "tif")]

# Rejoin them and reset the index
df = pd.concat([df_tif, df_other])
df.reset_index(drop=True, inplace=True)

# Creates a temporary column where every character is lowercase so it's easier for the remainder.
df["temp"] = df["File_name"].str.lower()

#############################################################################
#                                                                           #
# File name cleaning if needed before splitting to extract file names       #
#                                                                           #
#############################################################################

# Remove common sets of spacing characters and replace them with underscores
df["temp"] = clean_temp(df["temp"])

# Sort by bone, portion, and side.
df.sort_values(by=["Bone", "Portion", "Side"], inplace=True)

# Split the dataframe by location
df2 = df["Location"].str.split("\\", expand=True)

# Join together everything from the third row to the left
df2["Individual"] = ["_".join(row.astype(str)) for row in df2[df2.columns[:3]].values]

df["Individual"] = df2["Individual"]

# Mask rows with NaN values
inc_df = df[df[["Bone", "Portion"]].isnull().any(axis=1)]

# Split the incomplete dataframe by the location to extract missing information
df2 = inc_df["Location"].str.split("\\", expand=True)
inc_df["Bone"] = df2[6]

# Provide specific information if needed
inc_df["Portion"].iloc[0:1] = "Prox"

# Upate by index
df2 = df
df2["Bone"].update(inc_df["Bone"])
df2["Portion"].update(inc_df["Portion"])

# When satisfied
df = df2

# Join together 3 colums with an undercore "_"
df3["temp"] = df3[0].str.cat(df3[[1, 2]], sep="_")

# Only copy over row if the first character doesn't match the _ value
df3["name"] = df3["temp"].apply(lambda x: "" if x[:1] == "_" else x)

# Only copy over row if the last character doesn't match the _ value
df3["name"] = df3["temp"].apply(lambda x: "" if x[:-1] == "_" else x)

# Use lambda to take the first character over if it matches "_" otherwise take the whole thing
df3["name"] = df3["temp"].apply(lambda x: x[1:] if x[:1] == "_" else x)

# Use lambda to take the last character over to the left if it matches "_" otherwise take the whole thing
df3["name"] = df3["temp"].apply(lambda x: x[:-1] if x[-1:] == "_" else x)

# Only copy over row if the first character doesn't match the _ value
df3["name"] = df3["temp"].apply(lambda x: "" if x[1:] == "_" else x)


# Find a string in the dataframe using list comprehension, which is faster
drp = df[["TESTING" in x for x in df["File_name"]]].index[0]

# Drop the items by the index of the matched partial strings
df = df.drop(df.index[[drp]])

# Reset the index so they're sequential again
df.reset_index(drop=True, inplace=True)

# Replace a string in a column with another value
df["temp"] = df["temp"].str.replace(", ", "_")

# This uses regex, so you need to escape the character before the elimination
df["temp"] = df["temp"].str.replace(re.escape(") l_"), re.escape(")_"))


#############################################################################
#                                                                           #
#                  Isolate the individual name                              #
#                                                                           #
#############################################################################

# Split by the spaces
df2 = df["temp"].str.split("_", expand=True)

df3 = df2["temp"].str.split("_", expand=True, n=1)
df2["temp"] = df3[1]

# If the split worked and isolated only the names join it back to the main dataframe
df["Individual"] = df2[0]

# Drop the temp column
df.drop(columns=["temp"], inplace=True)

# Get the length of a column so we can sort
df["length"] = df["Individual"].str.len()
df.sort_values(by=["length"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Cast the dataframe as an object and replace blank spaces with numpy NaN so you can join them easily
df4 = df3.astype(object).replace("", np.NaN)
df4["temp"] = np_concat(df4[0], df4[1], df4[2], df4[3], df4[4], df4[5], join_char="_")
df["Individual"] = df4["temp"]

#############################################################################
#                                                                           #
#                        Isolate shaft elements                             #
#                                                                           #
#############################################################################

# Pull out just the shafts
df2 = isolate_shafts(df, search_column="Location")

# Remove the shafts from the main dataframe by retaining those index items from df2 that aren't in the main dataframe.
df = df[~df.index.isin(df2.index)]

# Reset the index so they're sequential again
df.reset_index(drop=True, inplace=True)

df3 = df2["Location"].str.rsplit("\\", n=1, expand=True)

df2["temp"] = df3[1].str.lower()

# Creates a temporary column where every character is lowercase so it's easier for the remainder.
df2["temp"] = df2["temp"].str.lower()

# Remove common sets of spcing characters and replace them with underscores
df2["temp"] = clean_temp(df2["temp"])

# Clean it up until the can explode it by an underscore
df2["temp"] = df2["temp"].str.replace("jsnhmuk", "js_nhmuk")

df3 = df2["temp"].str.split("_", n=3, expand=True)
df3 = df3[3].str.rsplit("_", n=1, expand=True)

df2["temp"] = df3[0]

df2 = pre_explode_stacked(df2)

# Unstack and copy the data from the
shaft_df = explode(df2, ["temp"], split_on="_")

shaft_df = setup_bone_columns(shaft_df, match_location="temp", shafts=True)


#############################################################################
#                                                                           #
#                  Isolate stacked individual names                         #
#         If you need to split up individuals in multiple scans             #
#                                                                           #
#############################################################################

df2 = df["Location"].str.rsplit("\\", n=1, expand=True)

df["temp"] = df2[1]

df = setup_temp(df)

df2 = df["temp"].str.rsplit("_", n=1, expand=True)

df2 = df2[0].str.split("_", n=3, expand=True)

df["temp"] = df2[3].str.lower()

# Clean it up until the can explode it by an underscore
df["temp"] = df["temp"].str.replace("fc1025_", "")

df = pre_explode_stacked(df)

# Unstack and copy the data from the
stacked_df = explode(df, ["temp"], split_on="_")

stacked_df = setup_bone_columns(stacked_df, match_location="temp", shafts=False)

df = pd.concat([stacked_df, shaft_df])
df.reset_index(drop=True, inplace=True)


# Subset it by the portion you what to fix, & for and, | for or.
df2 = df[(df.Bone == "Humerus") | (df.Bone == "C7")]

string_match = "C7_P|P_C7"

df2 = df2[df2["File_name"].str.contains(string_match)]


# Create a dataframe that matches a certain string
remove = df[df["Portion"].str.contains(string_match)]

# Use the index of the new dataframe to subset the old one
new_df = df[~df.index.isin(remove.index)]

# Redo the index so it's numerical
df2.reset_index(drop=True, inplace=True)

# Create a new dataframe
df2["Bone"] = "Patella"

# Concatenate the old and new dataframes for a complete dataframe
df = pd.concat([df, df2], axis=0, sort=False)

# Recast the index
df.reset_index(drop=True, inplace=True)

#################################################
#                                               #
# More complex operations for multiple scans    #
#                                               #
#################################################

# Drop the backslashes, which requires two to escape the character in regex
df["Individual"] = df["Individual"].str.replace("\\", "")

# You can automate escaping characters with re.escape
df["Individual"] = df["Individual"].str.replace(re.escape("(a)"), "")
df["Individual"] = df["Individual"].str.replace(re.escape("(b)"), "")
df["Individual"] = df["Individual"].str.replace(re.escape("(c)"), "")
df["Individual"] = df["Individual"].str.replace(re.escape("(d)"), "")


# Removing all the parentheses
# df['Individual'] = df['Individual'].str.replace(r"\(.*\)","")

# Fixing a specific row in a column
# df.at[16, 'Individual'] = "d13-2014(a)_d03-2018(c)_d01-2011(d)"

string_match = "shaft|_shaft_"

# Using a pattern match and a does not contain logical operator
print_df(test[(test["File_name"].str.contains(string_match) & ~(test["Bone"] == "C7"))])

# assign a string to a column using the logic from above
test = test[(test["File_name"].str.contains(string_match) & ~(test["Bone"] == "C7"))]
test["Portion"] = "Overview"

new_df.update(test["Portion"])
new_df.sort_values(by=["Portion", "temp", "Bone"], inplace=True)

new_df.sort_values(by=["File_name", "Bone", "Side"], inplace=True)
# Recast the index
new_df.reset_index(drop=True, inplace=True)

# Subset just the longest columns
df3 = df.iloc[43:]
df3["temp"] = df3["File_name"]
df3["temp"] = df3["temp"].str.replace(re.escape("70kV_120uA_1fps"), "")
df3["temp"] = df3["temp"].str.replace(re.escape("1fa "), "_")
new_df = explode(df3, ["temp"], split_on="_")
new_df["Bone"] = new_df["Bone"].str.capitalize()

new_df["Bone"].iloc[1:4] = new_df["temp"].iloc[1:4]
new_df["Bone"].iloc[5:] = new_df["temp"].iloc[5:]

new_df["Individual"].iloc[1:4] = new_df["Individual"].str.replace(
    re.escape("a3450"), "a3450_pt1"
)
new_df["Individual"].iloc[5:] = new_df["Individual"].str.replace(
    re.escape("a3450"), "a3450_pt2"
)
new_df = new_df.drop(index=[0, 4], inplace=False)
df = pd.concat([df, new_df])

df = df.drop(index=[86, 87], inplace=False)

df = pd.concat([df, new_df])


# Unstack and copy the data from the
new_df = explode(df3, ["Individual"], split_on="_")

# Drop the rows containing the long items using range
df.drop(index=list(range(29, 37)), inplace=True)

# Assign the full scans to the overview
new_df["Portion"] = "Overview"

# Concatenate the old and new dataframes for a complete dataframe
df2 = pd.concat([df, new_df], axis=0, sort=False)

df2.reset_index(drop=True, inplace=True)

# If there are no other assocaited numbers set this column to blank. Otherwise the next section is helpful.
df["Burial_Part_Accession"] = ""

# Get rid of Nan
df["Side"] = df["Side"].astype(str).replace("nan", "")

######################################
#                                    #
#                                    #
#  Any additional associated numbers #
#                                    #
#                                    #
######################################

# If there are additional numbers associated with the individuals
df_split = df2["Individual"].str.split("-", expand=True)

# Assign values to a new column based on another
# Here, where the length of the string is 2 the new column becomes 20 + the short string, and if  it isn't 2 then it copies over the longer string
df_split["Burial_Part_Accession"] = np.where(
    df_split[1].str.len() == 2, "20" + df_split[1], df_split[1]
)

# If needed, split the
df2["Burial_Part_Accession"] = df_split["Burial_Part_Accession"]

df2["Individual"] = df_split[0]


###################################
#                                 #
#      Checking for unique       #
#                                 #
###################################

# Check for duplicates before commitng
print(
    df[
        df.duplicated(
            subset=[
                "Individual",
                "Burial_Part_Accession",
                "Bone",
                "Portion",
                "Side",
                "Location",
            ],
            keep=False,
        )
    ]
)

print(
    df[
        df.duplicated(
            subset=["Individual", "Burial_Part_Accession", "Bone", "Portion", "Side"],
            keep=False,
        )
    ]
)
print(
    df[
        df.duplicated(
            subset=[
                "Individual",
                "Burial_Part_Accession",
                "Bone",
                "Portion",
                "Side",
                "File_type",
            ],
            keep=False,
        )
    ]
)

# Subset the duplicates
df1 = df[
    df.duplicated(
        subset=["Individual", "Burial_Part_Accession", "Bone", "Portion", "Side"],
        keep=False,
    )
]
df1.sort_values(by=["Individual", "Bone", "Portion"], inplace=True)

df1 = df1[
    [
        "Individual",
        "Burial_Part_Accession",
        "Bone",
        "Portion",
        "Side",
        "Location",
        "File_name",
        "File_type",
    ]
]


###
#
# Drop legitimate duplicates
#
###


# Removing actual duplicates
omit = [75, 27]

# Then drop this from the full dataframe
df = df.drop(index=omit, inplace=False)

df.reset_index(drop=True, inplace=True)


###
#
# Bone reassignments
#
###

C7_list = []
reassign_cell(pandas_df=df, column_name="Bone", location_list=C7_list, assingment="C7")

tibia_list = []
reassign_cell(
    pandas_df=df, column_name="Bone", location_list=tibia_list, assingment="Tibia"
)

femur_list = []
reassign_cell(
    pandas_df=df, column_name="Bone", location_list=femur_list, assingment="Femur"
)

humerus_list = []
reassign_cell(
    pandas_df=df, column_name="Bone", location_list=humerus_list, assingment="Humerus"
)


###
#
# Portion reassignments
#
###

dist_list = []
reassign_cell(
    pandas_df=df, column_name="Portion", location_list=dist_list, assingment="Dist"
)

prox_list = []
reassign_cell(
    pandas_df=df, column_name="Portion", location_list=prox_list, assingment="Prox"
)

whole_list = []
reassign_cell(
    pandas_df=df, column_name="Portion", location_list=whole_list, assingment="Whole"
)


shaft_prox_list = []
reassign_cell(
    pandas_df=df,
    column_name="Portion",
    location_list=shaft_prox_list,
    assingment="ShaftProx",
)

shaft_dist_list = []
reassign_cell(
    pandas_df=df,
    column_name="Portion",
    location_list=shaft_dist_list,
    assingment="ShaftDist",
)

shaft_mid_list = []
reassign_cell(
    pandas_df=df,
    column_name="Portion",
    location_list=shaft_mid_list,
    assingment="ShaftMid",
)

overview_list = []
reassign_cell(
    pandas_df=df,
    column_name="Portion",
    location_list=overview_list,
    assingment="Overview",
)

###
#
# Side reassignments
#
###

none_list = []

reassign_cell(pandas_df=df, column_name="Side", location_list=none_list, assingment="")

right_list = []

reassign_cell(
    pandas_df=df, column_name="Side", location_list=right_list, assingment="R"
)

left_list = [249]

reassign_cell(pandas_df=df, column_name="Side", location_list=left_list, assingment="L")


# Get unique instances of a value
df1["File_type"].value_counts()

# Assign a value to the file types
value_pattern = "vol|raw|tif|dcm"
df1["Value"] = (
    df1["File_type"].str.lower().str.extract("(" + value_pattern + ")", expand=False)
)
value_replace = {"vol": "1", "raw": "1", "tif": "2", "dcm": "2"}
df1["Value"] = df1["Value"].map(value_replace)

# Then create a new dataframe
df_dup = df1[
    df1.duplicated(
        subset=[
            "Individual",
            "Burial_Part_Accession",
            "Bone",
            "Portion",
            "Side",
            "Value",
        ],
        keep=False,
    )
]

# Get the files that match this
drp = df_dup["File_type"].str.contains("tif", case=True, regex=True)

# Turn the index into a list
drp = list(drp.index)

# Then drop this from the full dataframe
df = df.drop(index=drp, inplace=False)
df.reset_index(drop=True, inplace=True)

# Check for unwanted items
drp = df_dup["temp"].str.contains("VOI|IMJ", case=True, regex=True)

# df2 = (df[df.duplicated(subset=['Individual', 'Burial_Part_Accession', 'Bone', 'Portion', 'Side'], keep="last")])
# df3 = (df[df.duplicated(subset=['Individual', 'Burial_Part_Accession', 'Bone', 'Portion', 'Side'], keep="first")])

df_dup = df1["Location"]
df_dup = df_dup.str.replace(re.escape("humeralhead"), "")
df_dup = df_dup.str.replace(re.escape("femurdistal"), "")
df_dup = df_dup.str.replace(re.escape("__"), "_")
df_dup = df_dup.str.split("\\", expand=True)

df1["Side"].update(df_dup["Side"])

df_dup.columns = ["Individual", "Burial_Part_Accession"]
df_dup["Burial_Part_Accession"] = df_dup["Burial_Part_Accession"].str.replace(
    re.escape("../medtool_plugins/CT_scan_metadata_readers"), "_"
)
df_dup["Individual"] = df_dup["Individual"].str.replace(re.escape("_"), "")
df_dup["Individual"] = df_dup["Individual"].str.replace(re.escape("2ndind"), "_2")
df1["Individual"].update(df_dup["Individual"])
df1["Burial_Part_Accession"].update(df_dup["Burial_Part_Accession"])
df2["Individual"].update(df1["Individual"])
df2["Burial_Part_Accession"].update(df1["Burial_Part_Accession"])


###################################
#                                 #
#      Set up the inventory       #
#                                 #
###################################

df.drop(columns=["length"], inplace=True)

df.sort_values(by=["Bone", "Individual", "Portion"], inplace=True)

df.sort_values(by=["Individual", "Bone", "Portion"], inplace=True)

# Set Individual to be in caps, if appropriate
df["Individual"] = df["Individual"].str.upper()

# Or in sentence case
df["Individual"] = df["Individual"].str.capitalize()

# Assign the initial values
df["Population"] = str(population)
df["Species"] = str(species)
df["KC"] = int(3)
df["kmeans"] = int(3)
df["miathresh"] = int(2)
df["probability"] = float(0.02)
df["pcinmesh"] = float(1.75)


# Reorder the columns and place it into another dataframe
df_final = df[
    [
        "Population",
        "Individual",
        "Burial_Part_Accession",
        "Bone",
        "Portion",
        "Side",
        "Species",
        "kmeans",
        "miathresh",
        "probability",
        "Location",
        "File_name",
        "File_type",
        "KC",
        "pcinmesh",
    ]
]

# Get read of any NAN values
df_final.fillna("", inplace=True)

# Write it out so it can be referenced later
df_final.to_csv(population + "_inventory.csv", index=False)

# Get an initial par file generated from the data
par = get_par(df_final)

par.to_csv(population + ".par", index=False, sep=";")

# Convert the windows par to linux
base_folder = pathlib.Path(r"Z:\RyanLab")
linux_base_folder = pathlib.Path(
    r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab"
)

# Turn it into a linux par and write it out
linux_par = linuxify_par(par, base_folder, linux_base_folder)
linux_par.to_csv(population + "_linux.par", index=False, sep=";")


###################################
#                                 #
#      Adjust an  inventory       #
#                                 #
###################################

# define the population
population = "Tec_P"

# Read in the csv
df = pd.read_csv(str(population) + "_inventory.csv")

# Subset by a partial string match
df2 = df2[df2[8].str.contains("um")]

# If there are no other assocaited numbers set this column to blank. Otherwise the next section is helpful.
df["Burial_Part_Accession"] = ""

# Get rid of the nan values
df = df.replace(np.nan, "")

# Reorder the columns and place it into another dataframe
df_final = df[
    [
        "Population",
        "Individual",
        "Burial_Part_Accession",
        "Bone",
        "Portion",
        "Side",
        "Species",
        "kmeans",
        "miathresh",
        "probability",
        "Location",
        "File_name",
        "File_type",
        "KC",
    ]
]

# Get an initial par file generated from the data
par = get_par(df_final)

# Here, where the length of the string is 2 the new column becomes 20 + the short string, and if  it isn't 2 then it copies over the longer string
# df["$oldname"] = np.where(test["$oldname"].str[:-1] == "_", test["$oldname"].str[1].replace"")

df3["name"] = np.where(df3["temp"].str[:1] == "_", df3["temp"].str[:1].replace("_", ""))

df_split["Burial_Part_Accession"] = np.where(
    df_split[1].str.len() == 2, "20" + df_split[1], df_split[1]
)

df["Individual"] = df["Individual"].str.replace(re.escape("(a)"), "")

# Only copy over row if the first character doesn't match the _ value
df3["name"] = df3["temp"].apply(lambda x: "" if x[:1] == "_" else x)

df3["name"] = df3["temp"].apply(lambda x: x[:1].replace("_", "") if x[:1] == "_" else x)
df3["temp"].str[:1].replace("_", "")
# Write it out so it can be referenced later
df_final.to_csv(population + "_inventory.csv", index=False)


par["$oldname"] = par["$oldname"].astype(str).str[:-1].astype(str)

# Write out the parfile
par.to_csv(population + ".par", index=False, sep=";")

# Get rid of the nan values

# Convert the windows par to linux
base_folder = pathlib.Path(r"Z:\RyanLab\Projects")
# linux_base_folder = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/")
linux_base_folder = pathlib.Path(r"/mnt/ics/RyanLab/Projects")

# Turn it into a linux par and write it out
linux_par = linuxify_par(par, base_folder, linux_base_folder)
linux_par.to_csv(population + "_linux.par", index=False, sep=";")
