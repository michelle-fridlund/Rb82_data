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
#from PIL import Image
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

# Plot each slice
def plot_slices(img, save_dir):
    (a, b, c) = img.shape
    for i in range(0, c):
        im = plt.imshow(img[:, :, i], cmap='plasma')
        plt.axis('off')
        plt.colorbar(im, label='MBq/L')
        plt.savefig(f'{save_dir}/{i}')
        plt.close("all")


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    parser.add_argument('--data_path', dest='data_path', type=str,
                        help="directory containing patient files",
                        required=True)
    # Specify a pkl file for list of patients
    parser.add_argument('--pkl_path', dest='pkl_path', help="pickle filepath")

    # Read arguments from the command line
    args = parser.parse_args()

    sort_gates(args)