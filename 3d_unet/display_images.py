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

# patient = '0941d97c-f7b0-425b-b8b6-fa2e6f6ca595'
# filename = '3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'
# # filename = '0941d97c-f7b0-425b-b8b6-fa2e6f6ca595_rest_predicted.nii.gz'

# im_path = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/{dose}_{mode}/{patient}/{filename}'
# # im_path = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/Rb82_denoise_e100_bz1_lr0.0001_k0_predicted/{patient}/{filename}'

# # im_path = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/Rb82_denoise_e100_bz1_lr0.0001_k0_predicted/{patient}/{filename}'
# # output = f'/homes/michellef/recon_im/{dose}_{mode.lower()}_stress_{patient}_norm_SUV'
# output = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/Rb82_denoise_e100_bz1_lr0.0001_k0_predicted/{patient}/images'


# Create output directory
def mkdir_(output):
    if not os.path.exists(output):
        os.makedirs(output)


def load_nifti(path):
    return [print(i) for i in glob.glob("{}/*.nii.gz".format(path), recursive=True)]


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
    if args.norm:
        return np.array(pixels/65535, dtype=np.dtype(d_type))  # ~ [0,1]
    if args.norm and args.suv:
        return np.array(pixels*80000/(65535*1149), dtype=np.dtype(d_type))  # SUV nromalised
    else:
        return np.array(pixels, dtype=np.dtype(d_type))


# matplotlib
def load_nib(args):
    # import nibabel as nib
    # import matplotlib.pyplot as plt

    # img = nib.load(im_path)
    # d_type = img.header.get_data_dtype()  # get data type from nifti header
    # img2 = normalise(np.array(img.get_fdata(), dtype=np.dtype(d_type)))
    # (a, b, c) = img2.shape

    # mkdir_(output)

    # # plot for all slices
    # for i in range(0, c):
    #     im = plt.imshow(img2[:, :, i], cmap='plasma')
    #     plt.axis('off')
    #     plt.colorbar(im, label='MBq/L')
    #     plt.savefig(f'{output}/{patient}_{i}')
    #     plt.close("all")

    # specific patient
    # im_path = find_patients(str(args.data))

    im_path = os.path.join(str(args.data), str(args.patient)) if args.patient else find_patients(str(args.data))

    if args.maxp:
        im_path = im_path[1:args.maxp+1]

    return im_path


# antspyx for entire .nii.gz file
def load_ants(im_path):
    import ants
    # img = ants.image_read(im_path)
    # ants.plot(img, filename=f'/homes/michellef/recon_im/{dose}_{mode.lower()}_rest_{patient}.png', cmap='plasma')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Add argument
    required_args.add_argument("--mode", "-m", dest='mode',  help="nib/ants", required=True)
    required_args.add_argument("--data", "-d", dest='data',  help="patient directory", required=True)
    # required_args.add_argument("--original", "-o", dest='original', type=bool,
    # help="original = True, predicted = False", required=True)

    # Image args
    parser.add_argument('--suv', dest='suv', type=bool,
                        default=True, help='normalise to suv: True/False')
    parser.add_argument('--norm', '-n', dest='norm', type=bool,
                        default=True, help='normalise: True/False')
    # Save args
    parser.add_argument('--patient', '-p', dest='patient', help='patient name')
    parser.add_argument('--filename', '-f', dest='filename',
                        default='3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz', help='file name')
    # For original only
    parser.add_argument('--dose', dest='dose', default='100', help='dose level')
    parser.add_argument('--state', dest='phase', default='stat', help='stat/dyn/ekg')
    parser.add_argument('--phase', dest='phase', default='rest', help='stat/dyn/ekg')

    parser.add_argument('--maxp', dest='maxp', type=int, help='maximum number of patient to process')

    # Read arguments from the command line
    args = parser.parse_args()

    dir_path = str(args.data)

    mode = str(args.mode)

    if mode == 'nib':
        load_nib(args)
    elif mode == 'ants':
        load_ants(dir_path)
    else:
        print('Wrong input!')

    print('Done.')
