# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:04:26 2021

@author: michellef

######################################
PREPARE Rb82 DATA
######################################
"""
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import os
import re
import glob
import numpy as np
from matplotlib import pyplot as plt
import pydicom
import pickle
import argparse 
from pathlib import Path
import nibabel as nib

def get_name(string, **name_): 
     if name_.get("regex") == "out": #Getting output directory
         return Path(string).parents[0]
     if name_.get("regex") == "name": #Getting patient name
         return os.path.basename(Path(string).parents[0])
     else:
         print('Sequence not found')

#Transform DICOMS into numpy
def load_dicom(data_path):
    slices = [pydicom.read_file(i) for i in glob.glob("{}/*.ima".format(data_path), recursive = True)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices

#Return stack of all slices
def dcm2numpy(data_path):
    slices = load_dicom(data_path)
    image = np.stack([s.pixel_array for s in slices])
    return np.array(image, dtype = np.float16)
    
    
def find_patients(data_path):
    patients = {}
    for (dirpath, dirnames, filenames) in os.walk(data_path): 
        dirname = str(Path(dirpath).relative_to(data_path)) 
        if '/REST' in dirname and '/Sinograms' not in dirname \
            or '/STRESS' in dirname and '/Sinograms' not in dirname:
                new_path = os.path.join(data_path, dirname)
                patient_name = get_name(new_path, regex = 'name')
                phase = os.path.basename(dirname)
                patients[f'{patient_name}_{phase}'] = str(new_path)
    return patients

def save_data(data_path):
        patients = find_patients(data_path)
        for k, v in patients.items():
            np_arr = dcm2numpy(v)
            np_ = np_arr.reshape(128,128,111)
            # output = get_name(v, regex = 'out')
            # np.save(os.path.join(output, f'{k}.npy'))

            

if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path with pdt files", required=True)
    #required_args.add_argument("--output", "-o", help="Sorted data output directory path", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = args.data
    #output_path = args.output

    if not os.path.exists(data_path):
        raise 'Data source directory does not exist'

    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    save_data(data_path)