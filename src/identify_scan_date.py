#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:02:09 2020

@author: michellef
"""
import os 
import re
from pathlib import Path 
from shutil import copyfile
import argparse


def check_dir(output):
    if not os.path.exists(output):
        os.makedirs(output)


def get_name(string, **name_):
    if name_.get("regex") == "date": #Getting date from DICOM header
       reg = re.search('(?<=\[)(.*)(?=\])', string)
       d = reg.group(0)
    elif name_.get("regex") == "txt": #When reading from text file
       reg = re.search('\]\s*(.*)', string)
       d = reg.group(1)
    else:
       reg = re.search('^(.*?)\/', string) #From dirname 
       d = reg.group(1)
    return d


#Copy files to destination
def copy_files(file_path, save_path):
    if not file_path.is_dir():
        return None

    for file in file_path.iterdir():
        src = os.path.join(file_path, os.path.basename(file))
        dst = os.path.join(save_path, os.path.basename(file))
        check_dir(dst)
        copyfile(src,dst)


#Find all CTs and read scan dates from DICOM header
def find_anon(dir_path): 
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
        dirname = str(Path(dirpath).relative_to(dir_path)) 
        if '/IMA' in str(dirname) or '/CT' in str(dirname): 
            new_path = os.path.join(dir_path, dirname) 
            os.chdir(new_path) 
            os.system(f"for d in * ; do echo strings | dcmdump $d \
                      --search StudyDate| (head -c 30; echo '{dirname}') \
                      >> /homes/michellef/anon.txt; break 1; done")

#Get folder basename and corresponding scan date from temporary        print(src + 'IS NOW IN' + dst) .txt
def sort_anon(dir_path,text_path):
    find_anon(dir_path)
    anon_patients = {}
    with open(text_path/'anon.txt') as f:
          for line in f.readlines():
              line_ = line.strip()
              name = get_name(line_, regex = 'txt')
              scan_date = get_name(line_, regex = 'date')[:4]
              print(scan_date)
              anon_patients[name] = scan_date
    os.remove(os.path.join(str(text_path),'anon.txt'))
    return anon_patients

#Define all the source and destination directories 
def copy_sorted_patients(dir_path, text_path):
    anonlist = sort_anon(dir_path,text_path)
    
    for key, date in anonlist.items():
        
        rest_dir = os.path.join(str(dir_path), get_name(key), 'REST')
        stress_dir = os.path.join(str(dir_path), get_name(key), 'STRESS')
        ct_dir = os.path.join(str(dir_path), key)
        
        save_dir_rest= os.path.join(str(dir_path)+'_JSReconReady', date, get_name(key), 'REST')
        save_dir_stress = os.path.join(str(dir_path)+'_JSReconReady', date, get_name(key), 'STRESS')
        
        save_dir_ct1 = os.path.join(str(dir_path)+'_JSReconReady', date, get_name(key), 'REST', os.path.basename(key))
        save_dir_ct2 = os.path.join(str(dir_path)+'_JSReconReady', date, get_name(key), 'STRESS', os.path.basename(key))

        copy_files(Path(ct_dir), Path(save_dir_ct1))
        copy_files(Path(ct_dir), Path(save_dir_ct2))
        
        copy_files(Path(rest_dir), Path(save_dir_rest))
        copy_files(Path(stress_dir), Path(save_dir_stress))


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)
    required_args.add_argument("--text", "-t", help="Temporary text file path", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    text_path = Path(args.text)

    # data_path = Path('/homes/michellef/Rb82/data/PET_OCT8_Anonymous')
    # text_path = Path('/homes/michellef')
    copy_sorted_patients(data_path, text_path)