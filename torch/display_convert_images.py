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
import numpy as np
import nibabel as nib
import pydicom
import shutil
import tqdm

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


def sort_gates(args):
    for i in range(1,112):
        src = f'{args.data_path}/PSFTOF-{i}.ima'
        save_path = create_save_dir(args.data_path)
        dst = f'{save_path}/PSFTOF-{i}.ima'
        print(dst)
        try:
            shutil.copy(src, dst)
            print(f"{i}. File copied successfully.")
        # For other errors
        except:
            print("Error occurred while copying file.")


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

    sort_gates(args)
    #np2dcm('/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/100p_STAT/3616f6a0-b08a-4253-b072-431f699f5886/REST', '/homes/michellef/my_projects/rhtorch/torch/rb82/inferences/3616f6a0-b08a-4253-b072-431f699f5886_rest/Inferred_LightningAE_ResUNET3D_newsplit_TIODataModule_bz4_128x128x16_k0_e600_e=506.nii.gz')