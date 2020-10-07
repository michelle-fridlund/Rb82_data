#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:06:56 2020

@author: michellef
"""

import os, shutil
from pathlib import Path
#CHECK FILES

def directory_check(output_dir):
    if os.getcwd() != output_dir:
        os.chdir(output_dir) 
        
        
def mkdir_(input_dir):
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        #print(f'We are at {dirpath}')
        for patient in dirnames:
            print(patient)
            try:
                output_dir1 = os.path.join(str(patient), 'SORTED/REST')
                output_dir2 = os.path.join(str(patient), 'STRESS')
                os.makedirs(output_dir1)  
                os.makedirs(output_dir2)
            except FileExistsError:
                print('Directories already exist')
        
# def find_LM(files, output_dir):
#     LM = []
#     for (dirpath, dirnames, filenames) in os.walk(current_path): 
#         filelist = glob.glob("{}/*LISTMODE*".format(dirpath), recursive = True)
     
     
     
     
#     input_dir = os.getcwd()
# #    files = os.listdir
#     for file in files:
#         new_path = shutil.move(f"{input_dir}/{file}", output_dir)

# def find_files(pt):
#     LM = [] 
#     if not pt.is_dir():
#         return None
#     for f in pt.iterdir():
#         if 'LISTMODE' in f.name:
#             LM.append(f)
#         if 'PHYSION' in f.name:
#             PH.append(f) 
#         if 'CALIBRATION' in f.name:
#             CL.append(f)
#     return LM, PH, CL

#def split_files(pt):
    

#SPLIT INTO REST and TEST, COPY CALIBRATION INTO BOTH



#IDNETIFY CORRECT CT AND COPY CT OVER

if __name__ == "__main__":
    
    pt = Path('/homes/michellef/Rb82/RAWDATA/')

    mkdir_(pt)
    