#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:48 2021


@author: michellef
"""

import argparse

dose = '10p'

patient = '0eef59dd-8192-4065-b3dc-05f6e00aa452'
filename = '3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'
output = f'/homes/michellef/recon_im/{dose}_stat_rest_{patient}'
im_path = f"/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/{dose}_STAT/{patient}/{filename}"

# patient = '0d64ef76-5f71-4485-b481-613f17beedfe_02'
# filename = f'{patient}_rest_predicted.nii.gz'
# output = f'/homes/michellef/recon_im/25p_stat_rest_predicted_{patient}'
# im_path = f'/homes/michellef/my_projects/rb82_data/Dicoms_OCT8//Rb82_denoise_e100_bz1_lr0.0001_k0_predicted/{patient}/{filename}'

# matplotlib
def load_nib(im_path):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt

    img = nib.load(im_path)
    img2 = np.array(img.get_fdata(), dtype='double')
    (a, b, c) = img2.shape
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
    ants.plot(img, filename=f'/homes/michellef/recon_im/{dose}_stat_rest_{patient}_colour.png', cmap='plasma')


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
