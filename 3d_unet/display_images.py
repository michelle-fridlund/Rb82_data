#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:48 2021


@author: michellef
"""

import numpy as np
import argparse
import os

dose = '25p'
mode = 'STAT'

patient = '0eef59dd-8192-4065-b3dc-05f6e00aa452'
# patient = '0ef7e890-6586-4876-a630-a3af8e7fd736'
filename = '3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'

output = f'/homes/michellef/recon_im/{dose}_{mode.lower()}_rest_{patient}_norm'
im_path = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/{dose}_{mode}/{patient}/{filename}'


def mkdir_(output):
    if not os.path.exists(output):
        os.makedirs(output)
        
# Normalise pixel values
def normalise(pixels):
    d_type = pixels.dtype
    return np.array(pixels/65535, dtype=np.dtype(d_type))
# matplotlib
def load_nib(im_path):
    import nibabel as nib
    import matplotlib.pyplot as plt

    img = nib.load(im_path)
    d_type = img.header.get_data_dtype()
    img2 = normalise(np.array(img.get_fdata(), dtype=np.dtype(d_type)))
    (a, b, c) = img2.shape
    
    mkdir_(output)
    # plot for all slices
    for i in range(0,c):
        plt.imshow(img2[:,:,i], cmap = 'plasma')
        plt.axis('off')
        plt.colorbar(label='MBq/L', anchor = True)
        plt.savefig(f'{output}/{patient}_{i}')
        plt.close("all")

# antspyx for entire .nii.gz file
def load_ants(im_path):
    import ants
    img = ants.image_read(im_path)
    ants.plot(img, filename=f'/homes/michellef/recon_im/{dose}_{mode.lower()}_rest_{patient}.png', cmap='plasma')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--mode", "-m", help="nib/ants", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    mode = str(args.mode)
    if mode == 'nib':
        load_nib(im_path)
    elif mode == 'ants':
        load_ants(im_path)
    else:
        print('Wrong input!')

    print('Done.')
