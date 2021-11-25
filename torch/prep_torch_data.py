# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:00:22 2021

@author: IFRI0015
"""

import os
import re
from shutil import copy, rmtree
from tqdm import tqdm
from pathlib import Path
import argparse
import dicom2nifti
import pre_process

FORCE_DELETE = False

save_path = '/homes/michellef/my_projects/rhtorch/torch/rb82/data'


def create_dir(output):
    if not os.path.exists(output):
        os.makedirs(output)


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
            patients[f'{patient_name}_{phase}'] = os.path.join(
                new_path, filename)
    return patients


# Create a dictionary where save dirname is key and full path to nii.gz is value
""" def find_patients(data_path):
    patients = {}
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        dirname = str(Path(dirpath).relative_to(data_path))
        if '/REST' in dirname and '/Sinograms' not in dirname and '/Gate' not in dirname \
                or '/STRESS' in dirname and '/Sinograms' not in dirname and '/Gate' not in dirname:
            new_path = str(Path(os.path.join(data_path, dirname)))
            patient_name = get_name(dirname, regex='name')
            phase = (get_name(dirname, regex='phase')).lower()
            filename = '3_psftof.nii.gz'
            patients[f'{patient_name}_{phase}'] = os.path.join(
                new_path, filename)
    return patients """


# Copy pet to new directory
def copy_pet(data_path):
    patients = find_patients(data_path)
    for k, v in tqdm(patients.items()):
        dst = os.path.join(save_path, k)
        print(v)
        try:
            copy(v, dst)
        except Exception as error:
            print(error)
            print(f'Cannot copy {v} to {dst}')
            continue
    print(f'{len(patients)} found...')


def rename_pet(data_path):
    patients = find_patients(data_path)
    for k, v in tqdm(patients.items()):
        dst = os.path.join(save_path, k)
        old = os.path.join(dst, os.path.basename(v))
        new = os.path.join(dst, 'pet_50p_stat.nii.gz')
        try:
            os.rename(old, new)
        except Exception as error:
            print(error)
            print(f'Cannot rename {old} to {new}')
            continue


def copy_ct(dir_path, ct_path):
    patients = find_patients(dir_path)  # pet path
    ct_name = '3_psftof.nii.gz'
    for k, v in tqdm(patients.items()):
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


# Gated images
def find_gates(dir_path):
    gates = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/Gate' in dirname:
            gates.append(dirname)
            # Optionally delete gates and corresponding nifti files
            if FORCE_DELETE:
                dst = str(Path(dirname).parent)
                new_path = os.path.join(dir_path, dirname)
                print(new_path)
                file = os.path.join(dir_path, dst, 'pet_100p_ekg.nii.gz')
                try:
                    #os.remove(file)
                    rmtree(new_path)
                except Exception as error:
                    print(error)
                    print(f'Cannot delete {file}')
                    continue
    return gates


def rename_gates(dir_path):
    gates = find_gates(dir_path)
    for gate in tqdm(gates):
        dst = str(Path(gate).parent)
        old = os.path.join(dir_path, dst, '3_psftof.nii.gz')
        new = os.path.join(dir_path, dst, 'pet_100p_ekg.nii.gz')
        try:
            os.rename(old, new)
        except Exception as error:
            print(error)
            print(f'Cannot rename {old} to {new}')
            continue


""" def convert_nii(dir_path):
    gates = find_gates(dir_path)
    print(gates)
    c = 1
    for gate in tqdm(gates):
        output_ = str(Path(gate).parent)
        print(os.path.join(output_))
        files = os.listdir(os.path.join(dir_path, gate))
        if len(files) == 0:
            print(f'!No files found for {gate}!')
        else:
            dicom_to_nifti(os.path.join(dir_path, gate),
                        os.path.join(dir_path, output_))
            #print(f'{c}. {gate} to {output_}')
        c += 1

    print(f'{c} patients converted.') """


def convert_nii(dir_path):
    patients = find_patients(dir_path)
    c = 1
    for k,v in patients.items():
        if FORCE_DELETE:
            try:
                os.remove(v)
                print(f'deleted {k}')
            except Exception as error:
                print(error)
                print(f'Cannot delete {v}')
                continue

        output_ = str(Path(v).parent)
        input_ = os.path.join(output_, 'STRESS')
 
        files = os.listdir(input_)

        if len(files) == 0 or len(files) != 112:
            print(f'!No files found for {input_}!')
        else:
            dicom_to_nifti(input_, output_)
            #print(f'{c}. {input_} to {output_}')
        c += 1

    print(f'{c} patients converted.')


def prep_nnunet(dir_path, nn_path):
    patients = os.listdir(dir_path)  # pet path

    c = 0
    for p in tqdm(patients):
        if 'pickle' not in str(p):
            src = os.path.join(dir_path, p, 'ct.nii.gz')
            dst = os.path.join(nn_path, 'ct.nii.gz')
        try:
            copy(src, dst)
        except Exception as error:
            print(error)
            print(f'Cannot copy {src} to {dst}')
            continue

        new_name = f'{nn_path}/{p}_{c}_0000.nii.gz'
        os.rename(dst, new_name)
        c += 1
    print('Done!')
    

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
    parser.add_argument(
        "--scale", dest='scale', type=float, help="Dose scale factor")

    # Read arguments from the command line
    args = parser.parse_args()

    #ct = str(args.ct)
    #pet = str(args.pet)
    data_path = str(args.data_path)
    #FORCE_DELETE = args.force

    #convert_nii(data_path)
    #copy_pet(data_path)
    #rename_pet(data_path)
    #rename_gates(data_path)

    processor = pre_process.Data_Preprocess(args)
    processor.load_data()

    #prep_nnunet(data_path, '/homes/michellef/my_projects/ct_thorax/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR')