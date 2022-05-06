import os
import numpy as np
import argparse
import nibabel as nib
from pathlib import Path
from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

req_files = os.listdir('/homes/claes/data_shared/Rb82_train/8cd25543-c065-11ec-b751-e3702ec34f99')


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

# Find patients with required files from predifined dir
def find_patients(args):
    data = args.data
    paths = {}
    patients = os.listdir(data) 
    for p in patients:
        path = os.path.join(data, p)
        if os.listdir(path) == req_files:
            # Return name and full path
            paths[p] = path
    return paths

# Split patients in train/valid
def data_split(args):
    paths = find_patients(args)
    patients = list(paths.keys())
    print(patients)
    print(f'{len(patients)} patients found')
    pts_train, pts_test = train_test_split(patients, test_size=0.2)
    #return pts_train, pts_test

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
    for k,v in data.items():
        for val in v['MAX']:
            iqrs.append(val)
    
    dpd = pd.Series(iqrs)
    # Median  
    median  = np.median(iqrs)

    # Histogram
    #dpd.plot.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    dpd.plot(style = '.', color='#607c8e')
    plt.axhline(y = median, color = 'r', linestyle = '-')
    plt.title('Maximal pixel intensity values')
    plt.xlabel('Patient')
    plt.ylabel('MAX I')
    plt.savefig('/homes/michellef/clinical_eval/HELLO3.png')
    plt.close("all")

    #df = pd.DataFrame.from_dict(data)


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
        data_split(args)