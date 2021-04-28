# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:27:20 2021

Calculate quantitative image metrics

@author: micellef
"""

import os
import glob
import argparse
import pickle
import numpy as np
import nibabel as nib
from math import log10, sqrt
import matplotlib.pyplot as plt
from pathlib import Path

folder = '3_rest-lm-00-psftof_000_000_ctmv_4i_21s'


def load_images(path):
    return [plt.imread(i) for i in glob.glob("{}/*.png".format(path), recursive=True)]

def find_nifti(path):
    return [nib.load(i) for i in glob.glob("{}/*.nii.gz".format(path), recursive=True)]

def normalise(img, **dose_):
    d_type = img.header.get_data_dtype()
    np_im = np.array(img.get_fdata(), dtype=np.dtype(d_type))
    if dose_.get("dose_") == "ld":
        return np.array(np_im*4/(65535), dtype=np.dtype(d_type))
    else:
        return np.array(np_im/(65535), dtype=np.dtype(d_type))
    

# Return PSNR as compared to respective target
def psnr_(hd, ld):
    mse = np.mean((hd - ld) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# Return normalsed RMSE as compared to respective target
def rmse_(hd, ld):
    rmse = sqrt(np.mean((hd - ld) ** 2))/np.mean(hd)
    return rmse


# Return structural similarity index, hd is target
def ssim_(hd, ld):
    from skimage.measure import compare_ssim as ssim
    svalue = ssim(hd, ld, multichannel=True)
    return svalue


# Return coefficient of variation
def cv_(im):
    return ((np.std(im))/(np.mean(im)))*100


# Return standard error on the mean
def err(value):
    return np.std(value)/np.sqrt(len(value))


# Create a dictionary where patients are keys with hd/ld values
def find_patients(args):
    patient_dict = {}
    patients = read_pickle(str(args.pkl_path))
    
    # Change this to {p}_rest/stress_predicted for inferecd
    for p in patients:
        if str(args.mode) == 'numpy':
            patient_dict[p] = {'hd': os.path.join(
                str(args.hd), p, folder), 'ld': os.path.join(str(args.ld), p, f'{p}_rest_predicted')}
    
        elif str(args.mode) == 'nifti':
            patient_dict[p] = {'hd': os.path.join(
                str(args.hd), p), 'ld': os.path.join(str(args.ld), p)}
            
    return patient_dict


# Create a dictionary with all metrics for low-dose directory
def get_metrics(args, hd_path, ld_path):

    metrics = {'psnr': [],
               'ssim': [],
               'nrmse': [],
               'cv_hd': [],
               'cv_ld': [], }
    
    if str(args.mode) == 'numpy':
        hd = load_images(hd_path)
        ld = load_images(ld_path)
    
    if str(args.mode) == 'nifti':
        hd = normalise(nib.load(os.path.join(str(hd_path), f'{folder}.nii.gz')))
        ld = normalise(nib.load(os.path.join(str(ld_path), f'{folder}.nii.gz')), dose_='ld')
        
        
    for im_hd, im_ld in zip(hd, ld):
        metrics['psnr'].append(psnr_(im_hd, im_ld))
        metrics['ssim'].append(ssim_(im_hd, im_ld))
        metrics['nrmse'].append(rmse_(im_hd, im_ld))
        metrics['cv_hd'].append(cv_(im_hd))
        metrics['cv_ld'].append(cv_(im_ld))

    return metrics


# Read test patient names from a pkl file
def read_pickle(pkl_file):
    summary = pickle.load(open('%s' % pkl_file, 'rb'))
    # Test patients are a list of a list
    return summary['test'][0]


# Write metrics into pkl
def build_pickle(args, hd_path, ld_path):
    metrics = get_metrics(args, hd_path, ld_path)
    # here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(ld_path, 'im_metrics_nifti.pickle'), 'wb') as p:
        pickle.dump(metrics, p)
    print('.')

# Extract info from pkl
def get_stats(p):
    psnr = p['psnr']
    ssim = p['ssim']
    nrmse = p['nrmse']
    cv_hd = p['cv_hd']
    cv_ld = p['cv_ld']
    return psnr, ssim, nrmse, cv_hd, cv_ld


# Append individual metric values for all image slices for all patient keys
def evaluate_patients(args):
    patient_dict = find_patients(args)
    stats_dict = {}
    for k in patient_dict.keys():
        for v in patient_dict.values():
            total_metrics = {'psnr': [],
                             'ssim': [],
                             'nrmse': [],
                             'cv_hd': [],
                             'cv_ld': [], }
            # Check if pkl exists already
            # os.remove('%s/im_metrics_nifti.pickle' % v['ld'])
            if not (Path('%s/im_metrics_nifti.pickle' % v['ld'])).exists():
                build_pickle(args, v['hd'], v['ld'])
            p = pickle.load(open('%s/im_metrics_nifti.pickle' % v['ld'], 'rb'))
            # TODO: technically could populate an array here, don't need dict
            # and reduce by 1 fucntion
            psnr, ssim, nrmse, cv_hd, cv_ld = get_stats(p)
            total_metrics['psnr'].append(psnr)
            total_metrics['ssim'].append(ssim)
            total_metrics['nrmse'].append(nrmse)
            total_metrics['cv_hd'].append(cv_hd)
            total_metrics['cv_ld'].append(cv_ld)
        stats_dict[k] = total_metrics
    return stats_dict

# Extract overall


def overall_stats(args):
    psnr_ = []
    ssim_ = []
    nrmse_ = []
    cv_hd_ = []
    cv_ld_ = []
    stats_dict = evaluate_patients(args)
    for k, v in stats_dict.items():
        psnr = np.array(v['psnr'][0])
        ssim = np.array(v['ssim'][0])
        nrmse = np.array(v['nrmse'][0])
        cv_hd = np.array(v['cv_hd'][0])
        cv_ld = np.array(v['cv_ld'][0])
        for p in psnr:
            psnr_.append(p)
        for s in ssim:
            ssim_.append(s)
        for r in nrmse:
            nrmse_.append(r)
        for h in cv_hd:
            cv_hd_.append(h)
        for l in cv_ld:
            cv_ld_.append(l)
    print(f"PSNR value is: {np.mean(psnr):.4f} + {err(psnr):.4f}")
    print(f"SSIM value is: {np.mean(ssim):.4f} + {err(ssim):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse):.4f} + {err(nrmse):.4f}")
    print(f"CV in low-dose: {np.mean(cv_ld):.4f}% + {err(cv_ld):.4f}")
    # print(
    #     f"CV in target: {np.mean(cv_hd):.4f}% + {err(cv_hd):.4f}%; CV in low-dose: {np.mean(cv_ld):.4f}% + {err(cv_ld):.4f}")


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: output image type and data path
    required_args.add_argument(
        "--hd", dest='hd',  help="Path to target image directory", required=True)
    required_args.add_argument(
        "--ld", dest='ld',  help="Path to low-dose image directory", required=True)
    
    # Process in numpy or nifti format
    required_args.add_argument(
        "--mode", dest='mode',  help="numpy/nifti", required=True)

    # Specify a pkl file for list of patients
    parser.add_argument('--pkl', dest='pkl_path', help="pickle file path")
    

    # Read arguments from the command line
    args = parser.parse_args()

    hd_path = args.hd
    ld_path = args.ld
    

    if args.pkl_path:
        overall_stats(args)
    else:
        get_stats(hd_path, ld_path)

    print('Done.')
