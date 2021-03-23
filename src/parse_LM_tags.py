#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020
##############################################################################
Fetches unique tags from the LM and writes into pickle 
##############################################################################
@author: michellef
"""
import pickle
import os
import linecache
import argparse
from pathlib import Path


# Find listmode files
def find_LM(dir_path):
    pt = Path(dir_path)
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if 'LISTMODE' in f.name or '.LM.' in f.name:
            return f
    return None

# Fetch LM info
def get_dump(new_path):
    lm = find_LM(new_path)
    if lm:
        os.system(f'strings "{lm}" | tail -200 > "{new_path}"/dump.txt')


# Find patients
def find_patients(dir_path):
    patients = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        new_path = Path(os.path.join(dir_path, dirname))
        get_dump(new_path)
        if (new_path/'dump.txt').exists():
            with open(new_path/'dump.txt') as f:
                for i, line in enumerate(f.readlines()):
                    if line.startswith('PETCT'): #StudyInstance is before PETCT tag
                        prev_line = linecache.getline(f'{str(new_path)}/dump.txt', i).strip()
                        patients[dirname] = prev_line 
    return patients
            
# Write dump files
def write_pickle(dir_path):
    patients = find_patients(dir_path)
    print(f'{len(patients.keys())} PATIENTS FOUND!')
    os.chdir(dir_path)
    with open('paediatrics_studyuid.pickle', 'wb') as p:
        pickle.dump(patients, p)
        
    print(pickle.load(open('paediatrics_studyuid.pickle','rb')))




if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: LM data path
    required_args.add_argument("--data", "-d", dest='data',  help="patient directory", required=True)

    # Read arguments from the command line
    args = parser.parse_args()

    dir_path = Path(args.data)

    write_pickle(dir_path)
