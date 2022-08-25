#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:05:11 2020

@author: michellef
"""
# import pickle
from pathlib import Path
from shutil import copyfile, copytree, move, rmtree
import glob
import os
import re
import argparse

from cv2 import KeyPoint_overlap

def makedirs_(dest):
    if os.path.exists(dest) and len(os.listdir(dest) == 0):
        rmtree(dest)

    if not os.path.exists(dest):
        os.makedirs(dest)

# Replaced with move for efficiency 
def copy_file(input_dir, output_dir, filename):
    copyfile(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

def copy_dir(input_dir, output_dir):
    copytree(input_dir, output_dir)

def move_file(input_dir, output_dir, filename):
    move(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

def delete_file(input_dir, filename):
    os.remove(os.path.join(input_dir, filename))

# Find patients by tag
def find_cts(dir_path):
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
        dirname = str(Path(dirpath).relative_to(dir_path)) 
        if 'ACCT/' in str(dirname): 
            ct_path = os.path.join(dir_path, dirname)
    return ct_path


def order_files(patients, input_dir, output_dir):
    for patient_dir, types in patients.items():
        patient_input_dir = os.path.join(str(input_dir), str(patient_dir))
        ct_path = find_cts(patient_input_dir)
        #Output
        rest_output_dir = os.path.join(str(output_dir), str(patient_dir), 'REST')
        stress_output_dir = os.path.join(str(output_dir), str(patient_dir), 'STRESS')
        rest_output_ct = os.path.join(str(output_dir), str(patient_dir), 'REST', 'CT')
        stress_output_ct = os.path.join(str(output_dir), str(patient_dir), 'STRESS', 'CT')

        try:

            for type_name, files in types.items():
                if not files:
                    continue

                makedirs_(rest_output_dir)
                makedirs_(stress_output_dir)
                makedirs_(rest_output_ct)
                makedirs_(stress_output_ct)

                copy_dir(ct_path, rest_output_ct)
                copy_dir(ct_path, stress_output_ct)

                if type_name == 'LISTMODE' or type_name == 'PHYSIO':
                    sorted_files = sorted(files)
                    move_file(patient_input_dir, rest_output_dir, sorted_files[0])
                    move_file(patient_input_dir, stress_output_dir, sorted_files[1])
                elif type_name == 'CALIBRATION':
                    copy_file(patient_input_dir, rest_output_dir, files[0])
                    copy_file(patient_input_dir, stress_output_dir, files[0])

        except FileExistsError:
            print(f'Directory "{patient_dir}" already exists. Skipping...')
            pass


def find_files(dir_path):
    directories = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if dirpath == str(dir_path) or '/' in dirname:
            continue
        directories[dirname] = [os.path.basename(x) for x in glob.glob("{}/*.ptd".format(dirpath), recursive=True)]
    return directories


# def find_files_alternative(dir_path):
#     directories = {}
#     if not dir_path.is_dir():
#         return None

#     for cur_path in dir_path.iterdir():
#         if not cur_path.is_dir():
#           continue

#         dirname = str(cur_path.relative_to(dir_path))
#         if not directories.get(dirname):
#             directories[dirname] = []

#         for inner_path in cur_path.iterdir():
#             if not inner_path.is_file():
#                 continueprint(f'{p} is now in {new_path}')

#             filename = str(inner_path.relative_to(cur_path))
#             directories[dirname].append(filename)

#     pprint.pprint(directories)
#     return directories


def id_files(dir_path):
    dirlist = find_files(dir_path)
    patients = {}
#    occurencies = {}
    for key, filename in dirlist.items():
        patient = {
            'LISTMODE': [],
            'PHYSIO': [],
            'CALIBRATION': [],
            'OTHER': [],
        }

        patient_name = re.search('^[^0-9]*', key).group()
        # if occurencies.get(patient_name):
        #     occurencies[patient_name] = (occurencies[patient_name][0], occurencies[patient_name][1] + 1)
        # else:
        #     occurencies[patient_name] = (key, 1)

        # # print(patient_name, occurencies[patient_name])
        # if occurencies[patient_name][1] > 1:
        #     patients.pop(occurencies[patient_name][0], None)
        #     continue

        for item in filename:
            if 'LISTMODE' in item or '.LM.' in item:
                patient['LISTMODE'].append(item)
            elif 'PHYSIO' in item:
                patient['PHYSIO'].append(item)
            elif 'CALIBRATION' in item:
                patient['CALIBRATION'].append(item)
            elif 'FDG' in item:
                pass
            else:
                patient['OTHER'].append(item)

        #if len(patient['LISTMODE']) != 2 or len(patient['PHYSIO']) != 2 or len(patient['CALIBRATION']) != 1:
        #    continue

        patients[key] = patient

    return patients


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path with pdt files", required=True)
    required_args.add_argument("--output", "-o", help="Sorted data output directory path", required=True)
    parser.add_argument("-p", "--purge", help="delete OTHER .ptd file types", action="store_true")
    parser.add_argument("-m", "--mode", help="move/clean", type=str)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not os.path.exists(data_path):
        raise 'Data source directory does not exist'

    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)

    patients = id_files(data_path)
    order_files(patients, data_path, output_path, args)
