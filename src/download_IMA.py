#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:49:18 2020

@author: michellef
"""
import os
import re
import glob
import pickle
from shutil import copy
from pathlib import Path
import argparse

# def open_pickle(pickle_path):
#     my_patients = pickle.load(open('MYDATA_OCT8/scaninfo.pkl','rb'))
#     print(my_patients)
#     return my_patients


def check_dir(output):
    if not os.path.exists(output):
        os.makedirs(output)

def get_name(dirname):
    return re.search('^[^0-9]*', dirname).group()


def find_available_patient_dirs(dir_path):
    directories = {}
    if not dir_path.is_dir():
        return None
    for cur_path in dir_path.iterdir():
        if not cur_path.is_dir():
            continue
        
        dirname = str(cur_path.relative_to(dir_path))
        patient_name = get_name(dirname)
        directories[patient_name] = str(cur_path)
    return directories


def copy_ima_to_pet(file_path, patients):
    if not patients:
        print('No files to copy')
        return
    
    for (dirpath, dirnames, filenames) in os.walk(file_path):
        dirname = str(Path(dirpath).relative_to(file_path))
        patient_name = get_name(dirname)
        for f in glob.glob("{}/*.IMA".format(dirpath), recursive=True):
            if 'RB' in f and patient_name in patients.keys():
                dst = os.path.join(patients[patient_name],'IMA')
                check_dir(dst)
                copy(f, dst)


if __name__ == "__main__":
    # Initiate the parser

    parser = argparse.ArgumentParser()
    
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument("--pet", "-p", help="PET source directory path with patient names", required=True)
    required_args.add_argument("--ima", "-i", help="IMA source directory path with patient names", required=True)

    # # Read arguments from the command line
    args = parser.parse_args()
    dir_path = Path(args.pet)
    file_path = Path(args.ima)
    
    if not os.path.exists(dir_path) or not os.path.exists(file_path):
        raise 'PET or/and IMA source directories do not exist'
    
    # dir_path = Path('/homes/michellef/test')
    # file_path = Path('/homes/claes/projects/Lowdose_Rb82/RAWDATA')

    patients = find_available_patient_dirs(dir_path)
    copy_ima_to_pet(file_path, patients)

