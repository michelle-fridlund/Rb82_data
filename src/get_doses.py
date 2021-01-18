# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:11:27 2021

@author: IFRI0015
"""

import os
from pathlib import Path 
from datetime import datetime
import re
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from progress.bar import Bar
from time import sleep 
import argparse

#Applies regex and takes some custom arguments on demand
def get_name(string, **name_):
    if name_.get("regex") == "date": #Getting date from DICOM header
       return (re.search('(?<=\[)(.*)(?=\])', string)).group(0)
    if name_.get("regex") == "txt": #When reading from text file
       return (re.search('\]\s*(.*)', string)).group(1)
    if name_.get("regex") == "folder": #For JSRecon sorted folders
       return (re.search('(?<=\/)(.*)(\/)', string)).group(1)
    else:
       return (re.search('^(.*?)\/', string)).group(1) #From dirname (not year-sorted)
   
#Returns the first .ptd file in the parsed directory
def find_ima(pt):
    p = Path(pt)
    imas = []
    if not p.is_dir():
        return None
    for f in p.iterdir():
        if 'ima' in f.name:
            imas.append(f)
    return imas[0]

def check_dates(dir_path):
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
        dirname = str(Path(dirpath).relative_to(dir_path)) 
        if '/REST' in str(dirname) and '/Sinograms' not in str(dirname): 
            new_path = os.path.join(dir_path, dirname) 
            ima = find_ima(new_path)
            ima1 = os.path.basename(ima)
            name = get_name(dirname, regex = '')
            #print(os.path.basename(ima))
            os.chdir(new_path) 
            os.system(f"dcmdump {ima1} --search StudyDate | (head -c 30; echo '{name}') \
                      >> /homes/michellef/dates.txt")
            os.system(f"dcmdump {ima1} --search RadionuclideTotalDose | (head -c 30; echo '{name}') \
                      >> /homes/michellef/doses.txt")
                      
def find_dates(dir_path):
    #check_dates(dir_path)
    anon_patients = {}
    with open('/homes/michellef/dates.txt') as f:
          for line in tqdm(f.readlines(), desc ="Checking dates..."): #progress bar 
              line_ = line.strip()
              name = get_name(line_, regex = 'txt')
              #name = get_name(name1, regex='') #IF SORTED BY YEAR
              scan_date = get_name(line_, regex = 'date')
              anon_patients[name] = scan_date
              sleep(.001)
    return anon_patients
    
def write_obj(dir_path):
    out = open('/homes/michellef/test.txt', 'w')
    patient = find_dates(dir_path)
    for k, v in patient.items():
        out.write(v  + '\n')
    out.close()

if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    
    #find_dates(data_path)
    write_obj(data_path)