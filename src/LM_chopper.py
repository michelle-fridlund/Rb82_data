#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:23:30 2020

@author: michellef
"""

import os
import re
from pathlib import Path
from shutil import copyfile

def get_name(string, **name_):
    if name_.get("regex") == "date":
        return (re.search('(\/homes\/michellef\/Rb82\/data\/PET_OCT8_Anonymous_JSReconReady)\/(?<=\/)(.*)', string)).group(2)
    else:
        return (re.search('^(.*?)\/', string)).group(1)
    
#Return the second .ptd file
def find_LM(pt):
    p = Path(pt)
    ptds = []
    if not p.is_dir():
        return None
    for f in p.iterdir():
        if 'ptd' in f.name:
            ptds.append(f)
    return ptds[1]

#Find all listmodes
def find_files(dir_path):
    LM_list = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
            dirname = str(Path(dirpath).relative_to(dir_path)) 
            if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname): 
                    new_path = Path(os.path.join(dir_path, dirname))
                    name = get_name(str(new_path), regex = 'date')
                    ptds = find_LM(new_path)
                    LM_list[name] = str(ptds)
    return LM_list

def copy_file(input_dir, output_dir, filename):
    copyfile(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

def prep_chopper(dir_path):
    l = find_files(dir_path)
    #print(l)
    for k,v in l.items():
        #print(v)
        phase = os.path.basename(get_name(v, regex = 'ReconReady'))
        #new_name = f'{v}_{phase}'
        #print(f'{k}_{phase}')

find_files('/homes/michellef/Rb82/data/PET_OCT8_Anonymous_JSReconReady/2016')