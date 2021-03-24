#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:33:19 2020

@author: michellef
"""

import os
import re
import uuid
from pathlib import Path
from tqdm import tqdm
import argparse


def get_name(dirname):
    pt_name = re.findall('^[^0-9]*', dirname) 
    return str(pt_name)
    

def generate_random_name():
    return str(uuid.uuid4())


def find_patients(dir_path):
    pt_dict = {}
    if not dir_path.is_dir():
        return None
    
    for cur_path in dir_path.iterdir():
        dirname = str(cur_path.relative_to(dir_path))
        patient_name = get_name(dirname)
        pt_dict[patient_name] = [*pt_dict.get(patient_name, []), cur_path]
        #pt_dict[patient_name] = [pt_dict.extend(patient_name), cur_path]
    return pt_dict


def anon_pt(dir_path):
    patients = find_patients(dir_path)
    for patient_name, patient_path in tqdm(patients.items()):
        randomized_name = generate_random_name()
        for num, current_path in enumerate(patient_path, start=1):
            new_name = f'{randomized_name}_{num:02}' if len(patient_path) > 1 else randomized_name
            new_path = os.path.join(f'{dir_path}_anonymous', new_name)
            #print(f'{current_path} IS NOW {new_path}')
            #Call anonymize.exe via wine locally
            os.system(f'wine anonymize.exe -i {current_path} -o {new_path} -p -n {new_name}')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)

    anon_pt(data_path)
