import os
import re
import numpy as np
import argparse
import pickle
import nibabel as nib
from pathlib import Path
from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

req_files = os.listdir('/homes/claes/data_shared/Rb82_train/8cd25543-c065-11ec-b751-e3702ec34f99')

gated_files_ld = ['gate0_STRESS_25p.nii.gz', 'gate1_STRESS_25p.nii.gz', 'gate2_STRESS_25p.nii.gz', 'gate3_STRESS_25p.nii.gz', 'gate4_STRESS_25p.nii.gz', 'gate5_STRESS_25p.nii.gz', 'gate6_STRESS_25p.nii.gz', 'gate7_STRESS_25p.nii.gz', 'gate0_REST_25p.nii.gz', 'gate1_REST_25p.nii.gz', 'gate2_REST_25p.nii.gz', 'gate3_REST_25p.nii.gz', 'gate4_REST_25p.nii.gz', 'gate5_REST_25p.nii.gz', 'gate6_REST_25p.nii.gz', 'gate7_REST_25p.nii.gz']

gated_files_hd = ['gate0_STRESS_100p.nii.gz', 'gate1_STRESS_100p.nii.gz', 'gate2_STRESS_100p.nii.gz', 'gate3_STRESS_100p.nii.gz', 'gate4_STRESS_100p.nii.gz', 'gate5_STRESS_100p.nii.gz', 'gate6_STRESS_100p.nii.gz', 'gate7_STRESS_100p.nii.gz', 'gate0_REST_100p.nii.gz', 'gate1_REST_100p.nii.gz', 'gate2_REST_100p.nii.gz', 'gate3_REST_100p.nii.gz', 'gate4_REST_100p.nii.gz', 'gate5_REST_100p.nii.gz', 'gate6_REST_100p.nii.gz', 'gate7_REST_100p.nii.gz']


# Find all .nii.gz files in subdirs
def find_nifti(pt):
    pt = Path(pt)
    nifti = []
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if '.nii' in f.name:
            nifti.append(str(f))
    return nifti

# Find training patients with required files from predifined dir
def find_patients(args):
    data = args.data
    paths = {}
    patients = os.listdir(data) 
    for p in patients:
        path = os.path.join(data, p)
        # Check if patient dir contains all required the files
        result = all(files in os.listdir(path) for files in req_files)
        if result:
            # Return name and full path
            paths[p] = path
    return paths

# Split patients in train/valid
def data_split(args):
    paths = find_patients(args)
    patients = list(paths.keys())
    print(f'{len(patients)} patients found')
    pts_train, pts_test = train_test_split(patients, test_size=0.2)
    return pts_train, pts_test

# Write pickle file with data split
def write_pickle(args):
    pts_train, pts_test = data_split(args)

    data = {}
    data["train_0"] = []
    data["test_0"] = []
    data.update({"train_0": pts_train})
    data.update({"test_0": pts_test})

    with open('/homes/michellef/my_projects/rhtorch/torch/rubidium2022/data/rb82_final_train.pickle', 'wb') as p:
        pickle.dump(data, p)

# Nifti to numpy array
def nib2np(p):
    nifti = nib.load(p)
    d_type = nifti.header.get_data_dtype()  # get data type from nifti header
    pixels = np.array(nifti.get_fdata(), dtype=np.dtype(d_type))
    return pixels

# Calculate normalisation constant
def get_iqr(vals):
    return iqr(vals, rng=(0, 98))

# Calculate max intensity
def get_max(vals):
    return np.max(vals)

# Return a dict of individual IQR and MAX pixel values per patient
def get_stats(args):
    stat_dict = {}
    patients = find_patients(args)

    patient_num = len(list(patients.keys()))
    print(f'{patient_num} patients found')

    for name, path in patients.items():
        values = {
            'IQR': [],
            'MAX': [],
        }
        # Returns full path to file
        nifti = find_nifti(path)
        for n in nifti:
    # TODO: add image type as parser instread/
            if 'static_REST_100p' in n or 'static_STRESS_100p' in n:
            #if os.path.basename(n) in gated_files_ld: #n is a full path
                pixels = nib2np(n)
                iqr_value = get_iqr(pixels)
                max_value = get_max(pixels)
                values['IQR'].append(iqr_value)
                values['MAX'].append(max_value)

        stat_dict[name] = values

    return stat_dict

# Called with plot arg
def plot_values(args):
    data = get_stats(args)
    # Extract individual values - rewrite later
    iqrs =[]
    maxs = []

    for k,v in data.items():
        for val in v['IQR']:
            iqrs.append(val)
        for val in v['MAX']:
            maxs.append(val)

    
    dpd = pd.Series(iqrs)
    # Median  
    median_val  = np.median(iqrs)
    mean_val  = np.mean(iqrs)

    main_iqr = get_iqr(maxs)

    print("98th percentile of max SUV across training patients: {:.1f}".format(main_iqr))

    #out = open('/homes/michellef/clinical_eval/May9/iqr_static.txt', 'w')
    #out.write(f'Median: {median_val} \n')
    #out.write(f'Mean: {mean_val} \n')
    #out.close()

    # Histogram
    #dpd.plot.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    dpd.plot(style = '.', color='#607c8e')
    plt.axhline(y = median_val, color = 'r', linestyle = '-')
    plt.text(-130.0, median_val, "{:.1f}".format(median_val), horizontalalignment='left', size='medium', color='r', alpha = 0.8)
    plt.title('Gated IQR (0, 99.5) values for each patient (rest + stress)')
    #plt.title('Static maximal pixel values for each patient (rest + stress)')
    plt.xlabel('Patient #')
    plt.ylabel('IQR')
    #plt.ylabel('MAX I')
    plt.savefig('/homes/michellef/clinical_eval/May9/1_iqr_static_ld_99p.png')
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', type=str, 
                        default='/homes/claes/data_shared/Rb82_train/',
                        help='data source directory')
    parser.add_argument("-p", "--plot", help="increase output verbosity", 
                        action="store_true")

    args = parser.parse_args()

    if args.plot:
        plot_values(args)
    else:
        write_pickle(args)