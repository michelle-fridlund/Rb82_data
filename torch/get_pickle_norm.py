import os
import numpy as np
import argparse
import pickle
import nibabel as nib
from save_train_data import find_nifti
from tqdm import tqdm
from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

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

def write_pickle_test():
    patients = os.listdir('/homes/michellef/my_projects/rhtorch/static_ld')
    data = {}
    data["train_0"] = []
    data["test_0"] = []
    data.update({"test_0": patients})
    data.update({"train_0": patients})

    with open('/homes/michellef/my_projects/rhtorch/split_test.pickle', 'wb') as p:
        pickle.dump(data, p)
    data = pickle.load(open('/homes/michellef/my_projects/rhtorch/split_test.pickle', 'rb'))
    print(data)

# Nifti to numpy array
def nib2np(p):
    nifti = nib.load(p)
    d_type = nifti.header.get_data_dtype()  # get data type from nifti header
    pixels = np.array(nifti.get_fdata(), dtype=np.dtype(d_type))
    return pixels

# Calculate normalisation constant
def get_iqr(vals):
    return iqr(vals, rng=(0, 95))

# Calculate max intensity
def get_max(vals):
    return np.max(vals)

# Calculate max intensity
def get_mean(vals):
    return np.mean(vals)

# Return a dict of individual IQR and MAX pixel values per patient
def get_stats(args):
    #stat_dict = {}
    data_path = str(args.data)
    patients = os.listdir(data_path)

    for patient in tqdm(patients):
        full_path = os.path.join(data_path, patient)
        max_values=[]
        # Returns full path to file
        nifti = find_nifti(full_path)
        for n in nifti:
            if 'static_REST_100p' in n or 'static_STRESS_100p.nii.gz' in n:
                pixels = nib2np(n)
                # Max pixel intensity per patient image
                max_value = get_max(pixels)
                max_values.append(max_value)

    main_iqr = get_iqr(max_values)
    print(f'IQR (0,98): {main_iqr}')

"""
    print(f'IQR (0,98): main_iqr')
    for name, path in patients.items():
        values = {
            'IQR': [],
            'MAX': [],
            'MEAN': [],
        }
                iqr_value = get_iqr(pixels)
                max_value = get_max(pixels)
                mean_value = get_mean(pixels)
                values['IQR'].append(iqr_value)
                values['MAX'].append(max_value)
                values['MEAN'].append(mean_value)

        stat_dict[name] = values

    return stat_dict """

# Called with plot arg
def plot_values(args):
    data = get_stats(args)
    # Extract individual values - rewrite later
    iqrs =[]
    maxs = []
    means = []

    for k,v in data.items():
        for val in v['IQR']:
            iqrs.append(val)
        for val in v['MAX']:
            maxs.append(val)
        for val in v['MEAN']:
            means.append(val)

    
    dpd = pd.Series(maxs)
    # Median  
    median_val  = np.median(maxs)
    mean_val  = np.mean(maxs)

    main_iqr = get_iqr(maxs)

    #grand_mean = get_mean(means)
    #print("Grand mean static HD: {:.1f}".format(grand_mean))

    print("98th percentile of max SUV across training patients: {:.1f}".format(main_iqr2))

    # Histogram
    #dpd.plot.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    dpd.plot(style = '.', color='#607c8e')
    #plt.axhline(y = mean_val, color = 'r', linestyle = '-')
    plt.axhline(y = main_iqr2, color = 'r', linestyle = '-')
    #plt.text(1000.0, 1100000, "Mean:", horizontalalignment='left', size='small', color='r', alpha = 0.8)
    #plt.text(990.0, mean_val, "{:.1f}".format(mean_val), horizontalalignment='left', size='small', color='r', alpha = 0.8)
    plt.text(992.0, main_iqr2+50000, "iqr(0,98):", horizontalalignment='left', size='small', color='r', alpha = 0.8)
    plt.text(986.0, main_iqr2-50000, "{:.1f}".format(main_iqr2), horizontalalignment='left', size='small', color='r', alpha = 0.8)
    plt.title('25% maximal pixel intensities per patient scan (static rest + stress)')
    plt.xlabel('Patient #')
    #plt.ylabel('IQR')
    plt.ylabel('MAX intensity value (normalised)')
    plt.savefig('/homes/michellef/clinical_eval/static_ld_maxint_norm.png')
    plt.close("all")

    dpd.plot.box(color='#607c8e')
    plt.title('25% maximal pixel intensities per patient scan (static rest + stress)')
    plt.ylabel('MAX Intensity value (normalised)')
    plt.xlabel('patient image')
    plt.savefig('/homes/michellef/clinical_eval/static_ld_maxint_box_norm.png')
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help="plot statistics")
    parser.set_defaults(force=False)
    parser.add_argument('--data', dest='data', type=str, 
                        default='/homes/claes/data_shared/Rb82_train/',
                        help='data source directory')

    args = parser.parse_args()

    if args.plot:
        plot_values(args)
    else:
        get_stats(args)