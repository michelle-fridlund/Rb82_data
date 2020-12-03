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

def create_dir(output):
    if not os.path.exists(output):
        os.makedirs(output)

def get_name(string, **name_):
    if name_.get("regex") == "date":
        return (re.search('(\/homes\/michellef\/Rb82\/data\/PET_OCT8_Anonymous_JSReconReady)\/(?<=\/)(.*)', string)).group(2)
    if name_.get("regex") == "path":
        return (re.search('\/homes\/michellef\/(.*)', string)).group(1)
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

def LM_chopper(data_path, new_path):
    name = get_name(data_path, regex = 'path')
    my_dir = name.replace("/", "\\")
    string = f'cscript C:\\JSRecon12\\LMChopper64\\LMChopper64.js Z:\\{my_dir}'
    create_dir(new_path)
    os.chdir(new_path)
    f = open("run.bat", "w")
    # write line to output file
    f.write(string)
    f.close()
    #os.remove('run.bat')

def prep_chopper(dir_path):
    l = find_files(dir_path)
    for k,v in l.items():
        new_path = os.path.join('/homes/michellef/Rb82/data/PET_LMChopper_OCT8', k)
        LM_chopper(v,new_path)


prep_chopper('/homes/michellef/Rb82/data/PET_OCT8_Anonymous_JSReconReady/')