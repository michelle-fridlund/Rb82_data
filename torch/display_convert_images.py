#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 2021 15:29
@author: michellef
"""
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm 
import numpy as np
import nibabel as nib
import pydicom
import shutil

# Return all files of selected format in a directory
def find_files(dir_path, **file_format):
    if file_format.get("format") == "nifti":
        return [i for i in glob.glob("{}/*.nii.gz".format(dir_path),
                                     recursive=True)]
    elif file_format.get("format") == "numpy":
        return [i for i in glob.glob("{}/*.npy".format(dir_path),
                                     recursive=True)]
    elif file_format.get("format") == "ima":
        return [i for i in glob.glob("{}/*.ima".format(dir_path),
                                     recursive=True)]
    elif file_format.get("format") == "dcm" or \
            file_format.get("format") == "dicom":
        return [i for i in glob.glob("{}/*.dcm".format(dir_path),
                                     recursive=True)]
    else:
        print('Wrong format: nifti/numpy/ima/dicom.')


def create_save_dir(data_path, gate_number: int=1):
    save_path = f'{data_path}/Gate{gate_number}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


# Read pickle file
def read_pickle(pkl_path):
    summary = pickle.load(open('%s' % pkl_path, 'rb'))
    # Test patients (hardcoded)
    return summary['test_0']


# TODO: Rewrite this later for NOT hard-coded 
def denorm(pixels):
    return pixels*232429.9/4.0


# Return numpy array
def nifti2numpy(nifti):
    img = nib.load(nifti)
    d_type = img.header.get_data_dtype()  # get data type from nifti header
    # Network output is not in patient space???
    return denorm(np.array(img.get_fdata(), dtype=np.dtype(d_type)))


# Get a list of patients or read from pickle
def find_patients(args):
    patients = read_pickle(str(args.pkl_path)) if args.pkl_path \
               else os.listdir(args.data_path)
    print(patients)


# Get a list of patients or read from pickle
def dcm2numpy(d):
    d = pydicom.read_file(d)
    img = d.pixel_array.astype(np.float32)*d.RescaleSlope--d.RescaleIntercept
    return img


def plot_dicom(args):
    files = find_files(args.data_path, format='ima')
    save_dir = os.path.join(args.data_path, 'images')

    if len(files) == 0:
        print('No files found')

    sequence = []

 #   for f in files:
 #       dicom_raw = pydicom.dcmread(f)
 #       sequence.append(int(dicom_raw.SliceLocation))
 #   idx = [i[0] for i in sorted(enumerate(sequence), key=lambda s:s[1])]

    for i in range(1, 889):
        f = f'{args.data_path}{i}.ima'
        img = dcm2numpy(f)
        im = plt.imshow(img, cmap='plasma')
        plt.axis('off')
        plt.colorbar(im, label='MBq/L')
        plt.savefig(f'{save_dir}/{i}')
        plt.close("all")


# Convert nifti to dicom
def np2dcm(dicom_path, nifti_path):
    # Load all dicoms and corresponding nifti pixels
    dicom_files = find_files(dicom_path, format='ima')
    pixels = nifti2numpy(nifti_path)
    # Original DICOMS are int16
    pixels = pixels.astype(np.int16)
    # Create save dir in nifti dir
    saff = '/homes/michellef/my_projects/rhtorch/torch/rb82/inferences/3616f6a0-b08a-4253-b072-431f699f5886_rest'
    save_path = f'{saff}/DICOM'
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    (a, b, c) = pixels.shape
    for i in range (0, c):
        d = pydicom.read_file(dicom_files[i])
        pixel_data = pixels[:,:,i]
        d.PixelData = pixel_data.tostring()
        d.save_as(f'{save_path}/{i}.dcm')


# Sort cardiac gates with hard-coded dicom indeces       
def sort_gates(data_path):
    for i in range(778,889): # Make sure the gate number matches indeces!!!
        src = f'{data_path}/PSFTOF-{i}.ima'
        # TODO: User-defined gates and index ranges???
        # Copy selected dicoms over to user-defined gates 
        save_path = create_save_dir(data_path,gate_number = 8)
        dst = f'{save_path}/PSFTOF-{i}.ima'
        try:
            shutil.copy(src, dst)
            #print(f"{i}. File copied successfully.")
        # For other errors
        except:
            print("Error occurred while copying file.")


# Call gate sorting on selected patients
def find_patients(args):
    data_path = str(args.data_path)
    # Always start with the same index as last index of preceeding sequence
    patients = os.listdir(data_path)[153:173]
    for p in tqdm(patients):
        print(p)
        new_path = os.path.join(data_path, p, 'REST')
        new_path2 = os.path.join(data_path, p, 'STRESS')
        sort_gates(new_path)
        sort_gates(new_path2)
    # TODO: Run check for file length alongside sorting and output error if wrong
"""     for (dirpath, dirnames, filenames) in os.walk(data_path):
        dirname = str(Path(dirpath).relative_to(data_path))
        if '/REST' in dirname and '/Sinograms' not in dirname \
            or '/STRESS' in dirname and '/Sinograms' not in dirname:
            new_path = str(os.path.join(data_path, dirname))
            files = find_files(new_path, format='ima')
            if int(len(files)) != 888:
                print(dirname)
    print(f'{len(patients)} patients found...') """
    



if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    parser.add_argument('--data_path', dest='data_path', type=str,
                        help="directory containing patient files")
    # Specify a pkl file for list of patients
    parser.add_argument('--pkl_path', dest='pkl_path', help="pickle filepath")

    # Read arguments from the command line
    args = parser.parse_args()

    find_patients(args)