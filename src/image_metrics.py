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
from math import log10, sqrt
import matplotlib.pyplot as plt
from pathlib import Path


def load_images(path):
    return [plt.imread(i) for i in glob.glob("{}/*.png".format(path), recursive=True)]


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


# Create a dictionary with all metrics for low-dose directory
def get_metrics(hd_path, ld_path):

    metrics = {'psnr': [],
               'ssim': [],
               'nrmse': [],
               'cv_hd': [],
               'cv_ld': [], }

    hd = load_images(hd_path)
    ld = load_images(ld_path)
    print(f'{len(hd)} hd and {len(ld)} ld images found')

    for im_hd, im_ld in zip(hd, ld):
        metrics['psnr'].append(psnr_(im_hd, im_ld))
        metrics['ssim'].append(ssim_(im_hd, im_ld))
        metrics['nrmse'].append(rmse_(im_hd, im_ld))
        metrics['cv_hd'].append(cv_(im_hd))
        metrics['cv_ld'].append(cv_(im_ld))

    return metrics


# Write metrics into pkl
def build_pickle(hd_path, ld_path):
    metrics = get_metrics(hd_path, ld_path)
    os.chdir(ld_path)
    with open('im_metrics.pickle', 'wb') as p:
        pickle.dump(metrics, p)


# Extract info from pkl
def get_stats(hd_path, ld_path):
    print('ping')
    # Check if pkl exists already
    if not (Path('%s/im_metrics.pickle' % ld_path)).exists():
        build_pickle(hd_path, ld_path)
    print('pong')
    p = pickle.load(open('%s/im_metrics.pickle' % ld_path, 'rb'))

    psnr = p['psnr']
    ssim = p['ssim']
    nrmse = p['nrmse']
    cv_hd = p['cv_hd']
    cv_ld = p['cv_ld']
    print(f"PSNR value is: {np.mean(psnr):.4f} + {err(psnr):.4f}")
    print(f"SSIM value is: {np.mean(ssim):.4f} + {err(ssim):.4f}")
    print(f"NRMSE value is: {np.mean(nrmse):.4f} + {err(nrmse):.4f}")
    print(f"CV in target: {np.mean(cv_hd):.4f}% + {err(cv_hd):.4f}%; CV in low-dose: {np.mean(cv_ld):.4f}% + {err(cv_ld):.4f}")


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: output image type and data path
    required_args.add_argument("--hd", dest='hd',  help="Path to target image directory", required=True)
    required_args.add_argument("--ld", dest='ld',  help="Path to low-dose image directory", required=True)

    # Read arguments from the command line
    args = parser.parse_args()

    hd_path = args.hd
    ld_path = args.ld

    get_stats(hd_path, ld_path)

    print('Done.')
