#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:48 2021


@author: michellef
"""
import os
import numpy as np
import glob
import argparse
import nibabel as nib
from pathlib import Path

# Create parser


def parse_bool(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


# Create output directory
def mkdir_(output):
    if not os.path.exists(output):
        os.makedirs(output)


# Define save directory
def output_dir(args, i):
    # Create save dir from /homes/michellef/recon_im
    if args.original == True:
        rel_path = os.path.relpath(Path(i).with_suffix('').with_suffix(''), args.data)
        type_ = os.path.basename(Path(args.data))
        save_dir = f'/homes/michellef/recon_im/{type_}/{rel_path}'
    # Create save dir from filename (network output)
    else:
        save_dir = Path(i).with_suffix('').with_suffix('')
    return save_dir


def find_nifti(path):
    return [i for i in glob.glob("{}/*.nii.gz".format(path), recursive=True)]


def find_patients(dir_path):
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/REST' not in str(dirname) and '/STRESS' not in str(dirname) \
                and '/Sinograms' not in str(dirname):
            new_path = Path(os.path.join(dir_path, dirname))
            paths.append(new_path)
    return paths


# Normalise pixel values
def normalise(args, pixels):
    d_type = pixels.dtype
    if args.norm == True and args.suv == False:
        return np.array(pixels/65535, dtype=np.dtype(d_type))  # ~ [0,1]
    if args.suv == True and args.norm == False:
        return np.array(pixels*80000/(1149), dtype=np.dtype(d_type))  # SUV nromalised
    if args.norm == True and args.suv == True:
        return np.array(pixels*4*80000/(65535*1149), dtype=np.dtype(d_type))
    else:
        return np.array(pixels, dtype=np.dtype(d_type))


def load_patients(args):
    images = []
    # Specific patient
    im_path = [os.path.join(str(args.data), str(args.patient))] if args.patient else find_patients(str(args.data))

    # Limit number of patients
    if args.maxp:
        im_path = im_path[0:args.maxp+1]

    for p in im_path:
        im = find_nifti(p)
        if not len(im) == 0:
            for i in im:
                images.append(i)
    return images


# Matplotlib
def load_nib(args):
    import matplotlib.pyplot as plt

    images = load_patients(args)

    for i in images:
        img = nib.load(i)
        d_type = img.header.get_data_dtype()  # get data type from nifti header
        img2 = normalise(args, np.array(img.get_fdata(), dtype=np.dtype(d_type)))

        (a, b, c) = img2.shape

        save_dir = output_dir(args, i)
        print(save_dir)
        mkdir_(save_dir)

        # plot for all slices
        for i in range(0, c):
            im = plt.imshow(img2[:, :, i], cmap='plasma')
            plt.axis('off')
            plt.colorbar(im, label='MBq/L')
            plt.savefig(f'{save_dir}/{i}')
            plt.close("all")

# antspyx for entire .nii.gz file


def load_ants(args):
    import ants

    images = load_patients(args)

    for i in images:
        img = ants.image_read(i)

        # Create save dir in parent folder
        save_dir = output_dir(args, i)

        ants.plot(img, filename=f'{save_dir}.png', cmap='plasma')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: output image type and data path
    required_args.add_argument("--mode", "-m", dest='mode',  help="nib/ants", required=True)
    required_args.add_argument("--data", "-d", dest='data',  help="patient directory", required=True)
    # Changes save directory if niftis are not network output
    required_args.add_argument("--original", "-o", dest='original', type=parse_bool,
                               help="original (True)/predicted (False)", required=True)

    # Image args
    parser.add_argument('--suv', dest='suv', type=parse_bool, default=False,
                        help='normalise to suv: True/False')
    parser.add_argument('--norm', '-n', dest='norm', type=parse_bool, default=False,
                        help='normalise: True/False')

    # Choose single patient to process
    parser.add_argument('--patient', '-p', dest='patient', help='patient name')
    # Limit number of patient to process
    parser.add_argument('--maxp', dest='maxp', type=int, help='maximum number of patient to process')

    # Read arguments from the command line
    args = parser.parse_args()

    mode = str(args.mode)

    if mode == 'nib':
        load_nib(args)
    elif mode == 'ants':
        load_ants(args)
    else:
        print('Wrong input!')

    print('Done.')
