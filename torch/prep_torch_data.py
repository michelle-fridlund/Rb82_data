# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:00:22 2021

@author: IFRI0015
"""

import os
import re
from shutil import copy
from tqdm import tqdm
from pathlib import Path
import argparse
import dicom2nifti
import pre_process

FORCE_DELETE = False

save_path = '/homes/michellef/my_projects/rhtorch/rb82/data'

def get_name(string, **name_):
    if name_.get("regex") == "name":  # Getting date from DICOM header
        return (re.search('^(.*)\/', string)).group(1)
    elif name_.get("regex") == "phase":  # When reading from text file
        return (re.search('\/(.*)', string)).group(1)
    else:
        print('Unknown regex')


# Create a dictionary where save dirname is key and full path to nii.gz is value
def find_patients(data_path):
    patients = {}
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        dirname = str(Path(dirpath).relative_to(data_path))
        if '/REST' in dirname and '/Sinograms' not in dirname \
                or '/STRESS' in dirname and '/Sinograms' not in dirname:
            new_path = str(Path(os.path.join(data_path, dirname)).parent)
            patient_name = get_name(dirname, regex='name')
            phase = (get_name(dirname, regex='phase')).lower()
            filename = f'3_{phase}-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'          
            patients[f'{patient_name}_{phase}'] = os.path.join(new_path, filename)
    return patients


# Copy pet to new directory
def copy_pet(data_path):
    patients = find_patients(data_path)
    for k,v in tqdm(patients.items()):
        dst = os.path.join(save_path, k)
        try:
            copy(v, dst)
        except Exception as error:
            print(error)
            print(f'Cannot copy {v} to {dst}')
            continue
        
            
def rename_pet(data_path):
    patients = find_patients(data_path)
    for k,v in tqdm(patients.items()):
        dst = os.path.join(save_path, k)
        # old = os.path.join(dst, os.path.basename(v))
        old = os.path.join(dst, '5_ac_ct_cardiac_1.nii')
        new = os.path.join(dst, 'ct.nii.gz')
        try:
            os.rename(old,new)
        except Exception as error:
            print(error)
            print(f'Cannot rename {old} to {new}')
            continue
        

def copy_ct(dir_path, ct_path):
    patients = find_patients(dir_path) #pet path
    ct_name = '2_ac_ct_cardiac_1.nii.gz'
    for k,v in tqdm(patients.items()):
        dst = os.path.join(save_path, k, ct_name)
        name_ = k.split('_')[0]
        file_path = os.path.join(ct_path, name_, ct_name)
        print(file_path)
        try:
            copy(file_path, dst)
        except Exception as error:
            print(error)
            print(f'Cannot copy {file_path} to {dst}')
            continue
    

def dicom_to_nifti(input_, output_):
    if not os.path.exists(output_):
        os.makedirs(output_)
    dicom2nifti.convert_directory(input_, output_)


def find_cts(dir_path):
    cts = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/CT' in str(dirname) or '/IMA' in str(dirname):
            cts.append(dirname)
    return cts


def convert_nii(dir_path):
    cts = find_cts(dir_path)
    c = 1
    for ct in cts:
        output_ = str(Path(ct).parent)
        dicom_to_nifti(os.path.join(dir_path, ct),
                       os.path.join(dir_path, output_))
        print(f'{c}. {ct} to {output_}')
        c += 1

    print('Done.')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help="Force file deletion before copying")
    parser.set_defaults(force=False)

    required_args = parser.add_argument_group('required arguments')

    parser.add_argument(
        "--ct", dest='ct', help="data source directory")

    parser.add_argument(
        "--pet", dest='pet', help="data source directory")
    
    parser.add_argument(
        "--data_path", dest='data_path', help="nii.gz. data directory")
    parser.add_argument(
        "--norm", dest='norm', type=float, help="PET norm factor")
    
    # Read arguments from the command line
    args = parser.parse_args()

    ct = str(args.ct)
    pet = str(args.pet)
    
    rename_pet(pet)
    
    processor = pre_process.Data_Preprocess(args)
    processor.load_data()
