#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:07:16 2020

@author: michellef
"""
import numpy as np
import os
import argparse
import os
from re import match
import nibabel as nib
import glob
from pathlib import Path

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # DISABLE CPU WARNING
#    name = Path(file).stem
#    name = os.path.splitext(os.path.basename(file))[0]

#This is a test #

def getnib(file, dirpath):
    filepath = os.path.join(dirpath, file)
    try:
        nib_= nib.load(filepath)
    except FileNotFoundError:
        print(f'File {filepath} not found. Skipping...')
    return np.array(nib_.get_fdata(), dtype='double')

def mysave(file, dirpath):
    numpy = getnib(file,dirpath)
    name = Path(file).with_suffix('').with_suffix('')
    np.save(os.path.join(dirpath, str(name) + '.npy'), numpy)


def find_files(current_path):
    c = 0
    for (dirpath, dirnames, filenames) in os.walk(current_path):
        print(f'We are at {dirpath}')
        filelist = glob.glob("{}/*.nii.gz".format(dirpath), recursive = True)
        for file in filelist:
            mysave(file,dirpath)
            c += 1
    print('Done. %d numpy files saved.' %c)

if __name__ == "__main__":

    dirpath = '/homes/michellef/Amyloid/data_michelle/'
    find_files(dirpath)



