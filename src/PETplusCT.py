#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:06:28 2020

@author: michellef
"""
import os, shutil, re
from pathlib import Path

dir_path = '/homes/michellef/Rb82/retrieved_data'
output_dir = '/homes/michellef/test'

def get_name(dirname):
    pt_name = re.findall('^[^0-9]*', dirname) 
    name = pt_name[0]
    return name
    

for (dirpath, dirnames, filenames) in os.walk(dir_path):

    if 'CT' in str(dirpath) and len(filenames) == 111:
            d_path = str(Path(dirpath).relative_to(dir_path))
            new_dir = os.path.join(output_dir, d_path)
            shutil.copytree(dirpath, new_dir)
