#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:06:28 2020

@author: michellef
"""
import os, shutil, re
from pathlib import Path

dir_path = '/homes/michellef/test/retrieved_data'
pet_dist = '/homes/michellef/test/MYDATA'
output_dir = '/homes/michellef/test/MYCT'

def get_name(dirname):
    pt_name = re.findall('^[^0-9]*', dirname) 
    name = pt_name[0]
    return str(name)

def find_output(pet_dist):
    PET = []
    for PET_pt in os.listdir(pet_dist):
        PET_name = get_name(PET_pt)
        PET.append(PET_name)
    return PET

CT = []
PET = []
i = 0
for (dirpath, dirnames, filenames) in os.walk(dir_path):
    d_path = str(Path(dirpath).relative_to(dir_path))
    if 'CT' in str(dirpath) and len(filenames) == 111:
            CT_name = get_name(d_path)
            new_dir = os.path.join(output_dir, CT_name, f'CT{i}')
            shutil.copytree(dirpath, new_dir)
            i += 1
            # CT_patient = get_name(d_path)
            CT.append(CT_name)
            
for (dirpath, dirnames, filenames) in os.walk(pet_dist):
    d_path = str(Path(dirpath).relative_to(pet_dist))
    PET_name = get_name(d_path)
    PET.append(PET_name)
    for p in CT_name:
        if str(CT[p]) == str(PET[p]):
           new_dir = os.path.join(output_dir, str(PET[p])) 
           shutil.copytree(dirpath, new_dir)
    
    
            
# PET = []
# for (dirpath, dirnames, filenames) in os.walk(dir_path2):
#     dnames = os.listdir(dirpath2)
#     print(d_path)
#     PET_names = get_name(d_path)
#     PET.append(PET_names)
# print(PET)

