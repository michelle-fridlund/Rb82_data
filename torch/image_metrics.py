# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:13:25 2021

@author: IFRI0015
"""

import os
import argparse
import pickle
import numpy as np
import scipy
import nibabel as nib
from math import log10, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse


# Main helper parsers
def ParseLower(s):
    return s.lower()


def ParseBoolean(s):
    s = ParseLower(s)
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')

# Return standard error on the mean
def err(value):
    return np.std(value)/np.sqrt(len(value))


def psnr_(hd, ld):
    mse = np.mean((hd - ld) ** 2)
    if mse == 0:
        return "Same Image"
    return 10 * log10(np.amax(hd)**2/ mse)


# Return structural similarity index, hd is target
def ssim_(hd, ld):
    return structural_similarity(hd, ld, multichannel=True)


def rmse_(hd, ld):
    return normalized_root_mse(hd,ld,normalization = 'mean')
    # return sqrt(np.mean((hd - ld) ** 2))/np.mean(hd)

# Read test patient names from a pkl file
def read_pickle(pkl_file):
    summary = pickle.load(open('%s' % pkl_file, 'rb'))
    # Test patients are a list of a list
    train = len(summary['train_0'])
    test = len(summary['test_0'])
    #print(f'Train: {train}, Test: {test}')
    return summary['test_0']


def get_numpy(file_path):
    nifti = nib.load(file_path)
    # Get data type form nifti header
    d_type = nifti.header.get_data_dtype()
    return np.array(nifti.get_fdata(), dtype=np.dtype(d_type))

# Create a dictionary where patients are keys with hd/ld paths are values
def find_patients(args):
    patient_dict = {}
    patients = read_pickle(str(args.pkl_path))
    gate_num = args.gate_num
    hd = f'static_100p.nii.gz'
    ld = f'static_25p.nii.gz'
    out1 = f'LightningAE_ResNet_AE_bz8_1e-4_e300_TIODataModule_bz8_128x128x16_k0_e300.nii.gz'
    #out2 = f'LightningAE_ResNet_AE_bz8_1e-4_final_gate_static_TIODataModule_bz8_128x128x16_k0_e600_e297.nii.gz'
    #out3 = f'LightningAE_ResNet_AE_bz8_1e-4_final_gate_static_TIODataModule_bz8_128x128x16_k0_e600_e393.nii.gz'
    #out4 = f'LightningAE_ResNet_AE_bz8_1e-4_final_gate_static_TIODataModule_bz8_128x128x16_k0_e600_last.nii.gz'

    for p in patients:
        patient_dict[p] = {'hd': os.path.join(
            str(args.data), p, hd), 'ld': os.path.join(str(args.data), p, ld),
            'out1': os.path.join(str(args.inference), p, out1),
            #'out2': os.path.join(str(args.inference), p, out2),
            #'out3': os.path.join(str(args.inference), p, out3),
            #'out4': os.path.join(str(args.inference), p, out4),
            #'out5': os.path.join(str(args.inference), p, out5),
            }

    return patient_dict

# Create a dictionary with all metrics for low-dose directory
def get_metrics(args, **ld_type):

    patient_dict = find_patients(args)
    ld_type = str(ld_type.get("ld_type"))

    metrics = {'psnr': [],
               'ssim': [],
               'nrmse': [],}

    for k, v in patient_dict.items():
        hd = get_numpy(v['hd'])
        ld = get_numpy(v[ld_type])

        metrics['psnr'].append(psnr_(hd, ld))
        metrics['ssim'].append(ssim_(hd, ld))
        metrics['nrmse'].append(rmse_(hd, ld))

    return metrics

def return_values(dict_):
    psnr = dict_['psnr']
    ssim = dict_['ssim']
    nrmse = dict_['nrmse']
    return psnr, ssim, nrmse

# Extract info from pkl
def get_stats(args):

    metrics = get_metrics(args, ld_type = 'ld')
    metrics_inference1 = get_metrics(args, ld_type = 'out1')
    #metrics_inference2 = get_metrics(args, ld_type = 'out2')
    #etrics_inference3 = get_metrics(args, ld_type = 'out3')
    #metrics_inference4 = get_metrics(args, ld_type = 'out4')
    #metrics_inference5 = get_metrics(args, ld_type = 'out5')

    psnr, ssim, nrmse = return_values(metrics)
    psnr2, ssim2, nrmse2 = return_values(metrics_inference1)
    #psnr3, ssim3, nrmse3 = return_values(metrics_inference2)
    #psnr4, ssim4, nrmse4 = return_values(metrics_inference3)
    #psnr5, ssim5, nrmse5 = return_values(metrics_inference4)
    #psnr6, ssim6, nrmse6 = return_values(metrics_inference5)

    print('Original: \n')
    print(f"PSNR value is: {np.mean(psnr):.4f} + {err(psnr):.4f}")
    print(f"SSIM value is: {np.mean(ssim):.4f} + {err(ssim):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse):.4f} + {err(nrmse):.4f}")

    print('\n\n gated e241:')
    print(f"PSNR value is: {np.mean(psnr2):.4f} + {err(psnr2):.4f}")
    print(f"SSIM value is: {np.mean(ssim2):.4f} + {err(ssim2):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse2):.4f} + {err(nrmse2):.4f}")


    return metrics, metrics_inference1

"""
    print('\n\n gated e297:')
    print(f"PSNR value is: {np.mean(psnr3):.4f} + {err(psnr3):.4f}")
    print(f"SSIM value is: {np.mean(ssim3):.4f} + {err(ssim3):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse3):.4f} + {err(nrmse3):.4f}")

    print('\n\n gated e393:')
    print(f"PSNR value is: {np.mean(psnr4):.4f} + {err(psnr4):.4f}")
    print(f"SSIM value is: {np.mean(ssim4):.4f} + {err(ssim4):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse4):.4f} + {err(nrmse4):.4f}")

    print('\n\n gate last')
    print('Inference: ')
    print(f"PSNR value is: {np.mean(psnr5):.4f} + {err(psnr5):.4f}")
    print(f"SSIM value is: {np.mean(ssim5):.4f} + {err(ssim5):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse5):.4f} + {err(nrmse5):.4f}") 

    print('\n\n Gate + stat 1e-4 bz4')
    print('Inference: ')
    print(f"PSNR value is: {np.mean(psnr6):.4f} + {err(psnr6):.4f}")
    print(f"SSIM value is: {np.mean(ssim6):.4f} + {err(ssim6):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse6):.4f} + {err(nrmse6):.4f}") 

    diff1 = np.mean(psnr3) - np.mean(psnr)
    diff2 = np.mean(ssim3) - np.mean(ssim)
    diff3 = np.mean(nrmse3) - np.mean(nrmse)

    print(f"PSNR: {diff1}")
    print(f"SSIM: {diff2}")
    print(f"NRMSE: {diff3}")"""



def print_values(args):
    #gate_num = args.gate_num
    metrics, metrics_inference1 = get_stats(args)
    data = pd.DataFrame.from_dict(metrics)
    data1 = pd.DataFrame.from_dict(metrics_inference1)
    #data2 = pd.DataFrame.from_dict(metrics_inference2)
    
    
    concatenated = pd.concat([data.assign(image='Low-dose static'), data1.assign(image='Denoised low-dose static')])
    #concatenated = pd.concat([data.assign(image='Low-dose static'), data2.assign(image='Denoised low-dose')])
    ax = sns.boxplot(x=concatenated.image, y = concatenated.psnr, data = concatenated)
    plt.title(f'PSNR')
    #plt.xlabel('Image Type')
    #plt.ylabel('NRMSE')
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set(ylabel=None)
    #label_size = 100
    #mpl.rcParams['ytick.labelsize'] = label_size
    plt.yticks(fontsize = 15)
    plt.savefig(f'/homes/michellef/my_projects/rhtorch/AUG_19_static_psnr.png')

    df = pd.DataFrame(columns=['before','after',])
    for k,v in zip(metrics['psnr'], metrics_inference1['psnr']):
        df = df.append({'before': k, 'after':v},ignore_index=True)
    df.before = df.before.astype('float')
    df.after = df.after.astype('float')

    # Paired t-test
    from bioinfokit.analys import stat
    res = stat()
    res.ttest(df = df, res = ['after', 'before'], test_type = 3)
    print(res.summary)

if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: data path
    # Required args: inference path
    parser.add_argument(
        "--data", dest='data', default='/homes/michellef/my_projects/rhtorch/my-torch/rb82/data',
        help="Path to target image directory")
    required_args.add_argument(
        "--inference", dest='inference',  help="Path to inference directory", required=True)

    # Specify a pkl file for list of patients
    parser.add_argument('--pkl', dest='pkl_path', help="pickle file path",
        default = '/homes/michellef/my_projects/rhtorch/my-torch/rb82/data/rb82_final_train.pickle')
    # Select gate number
    parser.add_argument('--gate', dest='gate_num', type=int, help="Gate number")

    # Force delete old pickle files
    parser.add_argument('--delete', dest='delete', type=ParseBoolean,
                        default=False, help="purge existing info_pickle")

    # Read arguments from the command line
    args = parser.parse_args()

    #get_stats(args)
    print_values(args)
    print('Done.')
