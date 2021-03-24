#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:06:28 2020
##############################################################################
Copy CT to corresponding PET or create a new directory
##############################################################################
@author: michellef
"""
import os
import re
from shutil import copytree, rmtree
from tqdm import tqdm
from pathlib import Path
import pickle
import argparse

FORCE_DELETE = False

def parse_low(string):
    return string.lower()
    
# Return patient name from path
def get_name(dirname):
    return re.search('^[^0-9]*', dirname).group()


# Return patient names from pickle
def load_pickle(p_path):
    p = pickle.load(open(p_path, 'rb'))
    patients = p.keys()
    return patients


# Find patient dirs
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


# Copy directory
def copy_files(src, dst):
    try:
        # Remove existing trees (shutil won't copy otherwise)
        if FORCE_DELETE:
            rmtree(dst)
    except Exception:
        pass
    copytree(src, dst)


# Copy patients from pickle into a new directory
def copy_pet(pickle_path, pet_path):
    patients = load_pickle(pickle_path)
    save_path = f'{pet_path}_PETCT'
    for p in tqdm(patients):
        src = os.path.join(pet_path, p)
        dst = os.path.join(save_path, p)
        copy_files(src, dst)


# Copy CT to corresponding PET
def copy_ct_to_pet(dir_path, patients):
    if not patients:
        print('No files to copy')
        return

    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        patient_name = get_name(dirname)
        if 'CT' in dirname and patient_name in patients:
            try:
                dst = os.path.join(patients[patient_name], os.path.basename(dirpath))
                copy_files(dirpath, dst)
            except Exception as error:
                print(error)
                print(f'Cannot copy {dirname} to {dst}')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Force file deletion before copying")
    parser.set_defaults(force=False)
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--pet", "-p", help="PET source directory path with patient names", required=True)
    required_args.add_argument("--ct", "-c", help="CT source directory path with patient names", required=True)  
    required_args.add_argument("--mode", help="copy PET or CT directories",
                               type=parse_low, required=True)

    parser.add_argument('--pickle', dest='pickle_file', help='pickle file name')

    # Read arguments from the command line
    args = parser.parse_args()

    pet_path = Path(args.pet)
    ct_path = Path(args.ct)
    mode = str(args.mode)
    pickle_file = args.pickle_file
    FORCE_DELETE = args.force

    if not os.path.exists(pet_path) or not os.path.exists(ct_path):
        raise 'PET or/and CT source directories do not exist'
        
    if mode == 'ct':
        available_patients = find_available_patient_dirs(pet_path)
        copy_ct_to_pet(ct_path, available_patients)
    elif mode == 'pet':
        copy_pet(os.path.join(pet_path, pickle_file), pet_path)
    else:
        print('Copy PET or CT?')
