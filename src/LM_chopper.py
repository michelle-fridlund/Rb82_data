#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:23:30 2020

@author: michellef
"""
##############################################################################
#A script to prepare listmode files for LD-simulation with LMChopper64.js and 
#Copy the resulting correspong
import os
import re
from pathlib import Path
from shutil import copyfile
from progress.bar import Bar

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
def find_LM(pt, **name_):
    p = Path(pt)
    ptds = []
    if not p.is_dir():
        return None
    for f in p.iterdir():
        if 'ptd' in f.name:
            ptds.append(f)
    if name_.get("number") == "one":
        return ptds[1]
    else:   #Custom
        return ptds

#Find all listmodes
def find_files(dir_path):
    LM_list = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
            dirname = str(Path(dirpath).relative_to(dir_path)) 
            if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname): 
                    new_path = Path(os.path.join(dir_path, dirname))
                    name = get_name(str(new_path), regex = 'date')
                    ptds = find_LM(new_path, number = 'one')
                    LM_list[name] = str(ptds)
    return LM_list

#Prepare .bat executables for running LM chopper from petrecon 
#(Separately for REST & STRESS)
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

def delete_files(original_path):
    for (dirpath, dirnames, filenames) in os.walk(original_path): 
            dirname = str(Path(dirpath).relative_to(original_path)) 
            if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname): 
                    new_path = Path(os.path.join(original_path, dirname))
                    ptds = find_LM(new_path, number = '')
                    print(ptds[2]) #PLEASE MAKE SURE THE FILES ARE CORRECT FIRST!
                    #os.remove(ptds[2])
                    #os.chdir(str(new_path))
                    #os.remove('TempDicomHeader.IMA')

#Copy selected low dose into previously structured/copied folder
def copy_files(dir_path, dst):
    my_ptds = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
            dirname = str(Path(dirpath).relative_to(dir_path)) 
            if '/STRESS' in str(dirname): #can also add STRESS
                    new_path = Path(os.path.join(dir_path, dirname))
                    ptds = find_LM(new_path, number = '')
                    #Get simulated LD -->
                    #5p = [0], 10p = [1], 25p = [2], 50p = [3]
                    #my_ptds[dirname] = str(ptds[3])
                    if len(ptds) == 4:
                        my_ptds[dirname] = str(ptds[2])
                    else: 
                        print(f'{dirname} has {len(ptds)} files!!!')
                        pass
    #print(my_ptds)
    with Bar('Loading LISTMODE:', suffix='%(percent)d%%') as bar:
        for k,v in my_ptds.items(): #Add progress bar here
            save_path = os.path.join(dst, k)
            create_dir(save_path)
            copyfile(v, os.path.join(save_path, os.path.basename(v)))
            bar.next()
    print('Done!!!')
#TODO: Add arguments 

##Simulated data path
dir_path = '/homes/michellef/Rb82/data/PET_LMChopper_OCT8/2020'
##Temporary data path 
dst = '/homes/michellef/Rb82/data/PET_LMChopper_OCT8/2020_25p'

#ALSO USE THIS FOR DELETING ANY GIVEN DOSE LEVEL
#delete_files(dst) #Delete original LM file

##One at a time
copy_files(dir_path, dst)

#Use original listmode data path here
#prep_chopper('/homes/michellef/Rb82/data/PET_OCT8_Anonymous_JSReconReady/')