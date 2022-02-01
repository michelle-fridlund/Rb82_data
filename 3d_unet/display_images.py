#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:48 2021


@author: michellef
"""
import os
import re
import numpy as np
import glob
import argparse
import nibabel as nib
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


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
        rel_path = os.path.relpath(
            Path(i).with_suffix('').with_suffix(''), args.data)
        type_ = os.path.basename(Path(args.data))
        save_dir = f'/homes/michellef/recon_im/{type_}/{rel_path}'
    # Create save dir from filename (network output)
    else:
        save_dir = Path(i).with_suffix('').with_suffix('')
    return save_dir


# Return all nifti files in a directory
def find_nifti(path):
    return [i for i in glob.glob("{}/*gate8.nii.gz".format(path), recursive=True)]


# Read test patient names from a pkl file
def read_pickle(pkl_file):
    summary = pickle.load(open('%s' % pkl_file, 'rb'))
    # Test patients are alist of a list
    #return summary['test'][0]
    new_sum1 = summary['test_0']
    new_sum2 = summary['train_0']
    patients = []
    for p in new_sum1:
        patients.append(p)
    for p in new_sum2:
        patients.append(p)
    print(len(patients))
    return patients


def find_patients(args):
    paths = []
    dir_path = str(args.data)
    # Read from pickle
    if args.pkl_path:
        patients2 = read_pickle(str(args.pkl_path))
        patients = [p.split('_rest')[0] for p in sorted(patients2[::2])]
        for p in patients:
            paths.append(os.path.join(dir_path, p, 'REST'))
            paths.append(os.path.join(dir_path, p, 'STRESS'))
    # Find paths manually
    else:
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
    #60193.9
    if args.norm == True and args.suv == False:
        # ~ [0,1]
        return np.array(pixels/(232429.9), dtype=np.dtype(d_type))
    if args.suv == True and args.norm == False:
        # SUV nromalised
        return np.array(pixels*80000.0/(1149.0), dtype=np.dtype(d_type))
    if args.norm == True and args.suv == True:
        return np.array(pixels*80000/(232429.9*1149), dtype=np.dtype(d_type))
        # return np.array(pixels*4.0*80000.0/((232429.9*1149.0)), dtype=np.dtype(d_type))
    else:
        return np.array(pixels, dtype=np.dtype(d_type))


def load_patients(args):
    images = []
    # Specific patient
    im_path = [os.path.join(str(args.data), str(
        args.patient))] if args.patient else find_patients(args)
    # Limit number of patients
    if args.maxp:
        im_path = im_path[0:args.maxp]

    for p in im_path:
        im = find_nifti(p)
        #print(im)
        if not len(im) == 0:
            for i in im:
                images.append(i)
    return images


# Generate random hex colour codes
def gen_color(n):
    import random
    colors_list = []
    for i in range(0, n):
        color = "#"+''.join([random.choice('0123456789ABCDEF')
                             for j in range(6)])
        colors_list.append(color)
    return colors_list


def get_iqr(vals):
    from scipy.stats import iqr
    my_iqr = iqr(vals, rng=(25, 98))
    print(f'IQR value is: {my_iqr}')
    return my_iqr


# Plot and save individual slices
def plot_slices(img, save_dir):
    (a, b, c) = img.shape
    for i in range(0, c):
        im = plt.imshow(img[:, :, i], cmap='plasma')
        plt.axis('off')
        plt.colorbar(im, label='MBq/L')
        plt.savefig(f'{save_dir}/{i}')
        plt.close("all")


# Get stats/display with seaborn
def get_stats(args):
    im_dict = load_nib(args)
    keys = list(im_dict.keys())
    vals = [np.max(im_dict[k]) for k in keys]
    my_iqr = get_iqr(vals)
    print(len(keys))

    #s = sns.boxplot(vals)
    #s = sns.distplot(vals, label=keys, kde=False, bins=10)
    #s.set(xlim=(-1000, 100000))

    #colors_list = gen_color(len(keys))
    #plt.hist(vals, bins = 10, color = colors_list, label = keys)
    # plt.xlim([-1000,50000])
    #plt.legend(prop={'size':5})
    #plt.legend(['Maximal pixel values'])
    #plt.title('100% Static')
    #plt.xlabel('Maximal pixel intensity')
    #plt.savefig('/homes/michellef/100_stat.png')
    

# Plot and save individual images with matplotlib
def load_nib(args):
    images = load_patients(args)
    #print(images)
    im_dict = {}

    for i in images:
        img = nib.load(i)
        d_type = img.header.get_data_dtype()  # get data type from nifti header
        img2 = normalise(args, np.array(
            img.get_fdata(), dtype=np.dtype(d_type)))
        

        save_dir = output_dir(args, i)

        # create a numpy array dictionary per patient
        #name_ = (re.search('\STAT\/(.*?)-lm', i)).group(1)[0:4] + '...' \
        #    if args.original == True else i:
        name_ = i
        im_dict[name_] = img2

        if args.plot == True:
            print(f'Plotting {name_}')
            mkdir_(save_dir)
            plot_slices(img2, save_dir)

    return im_dict


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
    required_args.add_argument(
        "--mode", "-m", dest='mode',  help="nib/ants/stats", required=True)
    required_args.add_argument(
        "--data", "-d", dest='data',  help="patient directory", required=True)
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
    parser.add_argument('--maxp', dest='maxp', type=int,
                        help='maximum number of patient to process')

    # Specify a pkl file for list of testing patients
    parser.add_argument('--pkl', dest='pkl_path', help="dicom file directory")

    # Plot slices with matplotlib
    parser.add_argument('--plot', dest='plot', type=parse_bool, default=False,
                        help="matplotlib: True/False")

    # Read arguments from the command line
    args = parser.parse_args()

    mode = str(args.mode)

    if mode == 'stats':
        get_stats(args)
    elif mode == 'nib':
        load_nib(args)
    elif mode == 'ants':
        load_ants(args)
    else:
        print('Wrong input!')

    print('Done.')
