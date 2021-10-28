#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:23:30 2020

@author: michellef
"""
##############################################################################
# A script to prepare listmode files for LD-simulation with LMChopper64.js and
# Copy the resulting correspong
import os
import re
from pathlib import Path
from shutil import copyfile
from progress.bar import Bar
import argparse


def create_dir(output):
    if not os.path.exists(output):
        os.makedirs(output)


def get_name(string, **name_):
    if name_.get("regex") == "date":
        return (re.search('(\/homes\/michellef\/my_projects\/rb82_data\/PET_OCT8_Anonymous_JSReconReady)\/(?<=\/)(.*)', string)).group(2)
    if name_.get("regex") == "path":
        return (re.search('\/homes\/michellef\/(.*)', string)).group(1)
    else:
        return (re.search('^(.*?)\/', string)).group(1)


# Return the second .ptd file
def find_LM(pt, **name_):
    p = Path(pt)
    ptds = []
    if not p.is_dir():
        return None
    for f in p.iterdir():
        if 'ptd' in f.name:
            ptds.append(f)
    if len(ptds) != 0:
        if name_.get("number") == "one":
            return ptds[1]
        else:  # Custom
            return ptds


#Find paths
def get_paths(dir_path):
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                and 'quadratic' not in str(dirname) and '_25' not in str(dirname) \
                    and 'header' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                     and 'quadratic' not in str(dirname) and '_25' not in str(dirname) \
                         and 'header' not in str(dirname):
            new_path = Path(os.path.join(dir_path, dirname))
            paths.append(new_path)
    return paths


def find_files(dir_path):
    LM_list = {}
    paths = get_paths(dir_path)
    for new_path in paths:
        try:
            name = get_name(str(new_path), regex='date')
        except: 
            continue
        ptds = find_LM(new_path, number = 'one')
        LM_list[name] = str(ptds)
    return LM_list


# Prepare .bat executables for running LM chopper from petrecon
def LM_chopper(data_path, new_path):
    name = get_name(data_path, regex='path')
    my_dir = name.replace("/", "\\")

    string = f'cscript C:\\JSRecon12\\LMChopper64\\LMChopper64.js Z:\\{my_dir}'
    # create_dir(new_path)
    os.chdir(new_path)
    os.remove('run.bat')
    f = open("run.bat", "w")
    # write line to output file
    f.write(string)
    f.close()


def prep_chopper(dir_path):
    l = find_files(dir_path)
    for k, v in l.items():
        new_path = os.path.join('/homes/michellef/my_projects/rb82_data/PET_LMChopper_OCT8', k)
        LM_chopper(v, new_path)


def delete_files(dir_path):
    paths = get_paths(dir_path)
    for new_path in paths:
        #ptds = find_LM(new_path, number='')
        #for p in ptds:
        #    file = os.path.basename(str(p))
        #print(ptds[2]) # PLEASE MAKE SURE THE FILES ARE CORRECT FIRST!
        #os.remove(ptds[2])
        os.chdir(str(new_path))
        os.remove('run.bat')


# Copy selected low dose into previously structured/copied folder
def copy_files(dir_path, dst):
    my_ptds = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        # if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
        #         or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname):
        if '/REST_25' in str(dirname) or '/STRESS_25' in str(dirname):
            new_path = Path(os.path.join(dir_path, dirname))
            ptds = find_LM(new_path, number='')
            # Get simulated LD -->
            # 5p = [0], 10p = [1], 25p = [2], 50p = [3]
            dirname = str(Path(dirname).parents[0]) # Remove this later
            my_ptds[dirname] = str(ptds[0])
            # if len(ptds) == 4:
            #     my_ptds[dirname] = str(ptds[2])
            # else:
            #     print(f'{dirname} has {len(ptds)} files!!!')
            #     pass
        print(my_ptds.keys())
        with Bar('Loading LISTMODE:', suffix='%(percent)d%%') as bar:
            for k, v in my_ptds.items():  # Add progress bar here
                save_path = os.path.join(dst, k)
                create_dir(save_path)
                copyfile(v, os.path.join(save_path, os.path.basename(v)))
                print(v, os.path.join(save_path, os.path.basename(v)))
                bar.next()
    print('Done!!!')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--mode", "-m", help="delete/copy/prep", required=True)
    required_args.add_argument("--year", type=int, help="scan year", required=True)
    parser.add_argument('--force', action='store_true',
                        help="Force file deletion before copying")
    # Read arguments from the command line
    args = parser.parse_args()
    mode = str(args.mode)
    year = str(args.year)

    # TODO: Use relative paths
    # TODO: Read relative path from script arguments using an argument parser

    # Simulated data path
    dir_path = f'/homes/michellef/my_projects/rb82_data/PET_OCT8_Anonymous_JSReconReady/{year}'
    # Temporary data path
    dst = f'/homes/michellef/my_projects/rb82_data/PET_LMChopper_OCT8/{year}'

   # ALSO USE THIS FOR DELETING ANY GIVEN DOSE LEVEL
    if mode == 'delete':
        delete_files(dst)  # Delete original LM file
    # One at a time
    elif mode == 'copy':
        copy_files(dir_path, dst)
    # Use original listmode data path here
    else:
        # TODO: Use relative path
        # TODO: Read path from argument (maybe?)
        prep_chopper(dir_path)
