#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020

@author: michellef
"""
import pickle
import re
import os
from datetime import datetime
import argparse
from pathlib import Path


#Regex for fetching SeriesInstanceUID
def get_uid(string):
    return str(re.findall(r'(?<=UI8).*?(?=[\s])', string))


# Find listmode files
def find_LM(dir_path):
    pt = Path(dir_path)
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if 'LISTMODE' in f.name or '.LM.' in f.name:
            return f
    return None


def get_dump(new_path):
    lm = find_LM(new_path)
    # print(lm)
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
                for line in f.readlines():
                    # line_ = line.strip()
                    uid = get_uid(line)
                    patients[dirname] = uid
            # os.remove(new_path/'dump.txt')
    return patients
            
# Write dump files
def write_pickle(dir_path):
    patients =  find_patients(dir_path)
    os.chdir(dir_path)
    with open('series_uid.pickle', 'wb') as p:
        pickle.dump(patients, p)
        
    print(pickle.load(open('series_uid.pickle','rb')))




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
