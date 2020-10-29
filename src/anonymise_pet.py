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
#import string
#import random

# dir_path = Path('Z:\Rb82\PET_OCT8')
# anon_path = Path('Z:\Rb82\PET_OCT8_Anonymous')

dir_path = Path('/homes/michellef/Rb82/PET_OCT8')
anon_path = Path('/homes/michellef/Rb82/PET_OCT8_Anonymous')


def get_name(dirname):
    pt_name = re.findall('^[^0-9]*', dirname) 
    return str(pt_name)

# def random_name():
#     letters = string.ascii_letters
#     result_name = ''.join(random.choice(letters) for i in range(10))
#     return result_name
    
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
    for patient_name, patient_path in patients.items():
        randomized_name = generate_random_name()
        for num, current_path in enumerate(patient_path, start=1):
            new_name = f'{randomized_name}_{num:02}' if len(patient_path) > 1 else randomized_name
            new_path = os.path.join(str(anon_path), new_name)
            #print(f'{current_path} IS NOW {new_path}')
            os.system(f'wine anonymize.exe -i {current_path} -o {new_path} -p -n {new_name}')


anon_pt(dir_path)
