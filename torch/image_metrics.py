# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:13:25 2021

@author: IFRI0015
"""

import os
import argparse
import pickle
import numpy as np
import nibabel as nib
from math import log10, sqrt
import matplotlib.pyplot as plt
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
    return summary['test_0']


def get_numpy(file_path):
    nifti = nib.load(file_path)
    # Get data type form nifti header
    d_type = nifti.header.get_data_dtype()
    return np.array(nifti.get_fdata(), dtype='float64')
    
# Create a dictionary where patients are keys with hd/ld paths are values
def find_patients(args):
    patient_dict = {}
    patients = read_pickle(str(args.pkl_path))

    hd = f'pet_100p_stat_norm.nii.gz'
    ld = f'pet_25p_stat_norm.nii.gz'
    out = f'Inferred_LightningAE_UNET3D_v1_TIODataModule_bz4_128x128x16_k0_e600_e=600.nii.gz'


    for p in patients:
        patient_dict[p] = {'hd': os.path.join(
            str(args.data), p, hd), 'ld': os.path.join(str(args.data), p, ld),
            'out': os.path.join(str(args.inference), p, out)}
            
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

        metrics['psnr'].append(psnr_(hd,ld))
        metrics['ssim'].append(ssim_(hd,ld))
        metrics['nrmse'].append(rmse_(hd,ld))
    # for hd, ld in (load_nifti(patient_dict['hd']), load_nifti(patient_dict[f'{ld_type}'])):
    #     metrics['psnr'].append(psnr_(hd,ld))
    #     metrics['ssim'].append(ssim_(hd,ld))
    #     metrics['nrmse'].append(rmse_(hd,ld))
        
    return metrics

def return_values(dict_):
    psnr = dict_['psnr']
    ssim = dict_['ssim']
    nrmse = dict_['nrmse']
    return psnr, ssim, nrmse

# Extract info from pkl
def get_stats(args):
    
    metrics = get_metrics(args, ld_type = 'ld')
    metrics_inference = get_metrics(args, ld_type = 'out')
    
    psnr, ssim, nrmse = return_values(metrics)
    psnr2, ssim2, nrmse2 = return_values(metrics_inference)
    
    print('\n\n')
    print('Original: \n\n')
    print(f"PSNR value is: {np.mean(psnr):.4f} + {err(psnr):.4f}")
    print(f"SSIM value is: {np.mean(ssim):.4f} + {err(ssim):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse):.4f} + {err(nrmse):.4f}")
    
    print('\n\n')
    print('Inference: ')
    print(f"PSNR value is: {np.mean(psnr2):.4f} + {err(psnr2):.4f}")
    print(f"SSIM value is: {np.mean(ssim2):.4f} + {err(ssim2):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse2):.4f} + {err(nrmse2):.4f}")
    
    return psnr, ssim, nrmse



if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: data path
    required_args.add_argument(
        "--data", dest='data',  help="Path to target image directory", required=True)
    # Required args: inference path
    required_args.add_argument(
        "--inference", dest='inference',  help="Path to inference directory", required=True)

    # Specify a pkl file for list of patients
    parser.add_argument('--pkl', dest='pkl_path', help="pickle file path")

    # Force delete old pickle files
    parser.add_argument('--delete', dest='delete', type=ParseBoolean,
                        default=False, help="purge existing info_pickle")

    # Read arguments from the command line
    args = parser.parse_args()

    get_stats(args)

    print('Done.')