#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:06:28 2020

@author: michellef
"""
import os
import re
from shutil import copytree, rmtree
from pathlib import Path
import argparse

FORCE_DELETE = False


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


def copy_ct_to_pet(dir_path, patients):
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        patient_name = get_name(dirname)
        if '/CT' in dirname and len(filenames) == 111 and patient_name in patients:
            try:
                dst = os.path.join(patients[patient_name], os.path.basename(dirpath))
                try:
                    if FORCE_DELETE:
                        rmtree(dst)
                except Exception:
                    pass
                copytree(dirpath, dst)
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

    # Read arguments from the command line
    args = parser.parse_args()
    pet_path = Path(args.pet)
    ct_path = Path(args.ct)
    FORCE_DELETE = args.force

    if not os.path.exists(pet_path) or not os.path.exists(ct_path):
        raise 'PET or/and CT source directories do not exist'

    available_patients = find_available_patient_dirs(pet_path)
    copy_ct_to_pet(ct_path, available_patients)
