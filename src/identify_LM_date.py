#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:45:51 2020

##############################################################################
Script for comparing sorted patients' LISTMODE dates with corresponding CT 
ScanDate
##############################################################################

@author: michellef
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


#Returns the first .ptd file in the parsed directory
def find_LM(pt):
    p = Path(pt)
    ptds = []
    if not p.is_dir():
        return None
    for f in p.iterdir():
        if 'ptd' in f.name:
            ptds.append(f)
    return ptds[0]

#Writes the contents of the listmode files into dump.txt
def get_LM_date(pt):
    patients = find_LM(pt)
    if patients:
        os.system(f'strings {patients.absolute()} | tail -200 > {pt}/dump.txt')

#Applies regex and takes some custom arguments on demand
def get_name(string, **name_):
    if name_.get("regex") == "date": #Getting date from DICOM header
       return (re.search('(?<=\[)(.*)(?=\])', string)).group(0)
    if name_.get("regex") == "txt": #When reading from text file
       return (re.search('\]\s*(.*)', string)).group(1)
    if name_.get("regex") == "ReconReady": #For JSRecon sorted folders
       temp = (re.search('(\/homes\/michellef\/Rb82\/data\/PET_OCT8_Anonymous_JSReconReady)\/(?<=\/)(.*)', string)).group(2)
       return (re.search('(?<=\/)(.*)(\/)', str(temp))).group(1)
    else:
       return (re.search('^(.*?)\/', string)).group(1) #From dirname (not year-sorted)

#Goes into all LISMODE folders, fetches date 
def find_ptd(dir_path): 
    my_dates = {}
    with Bar('Loading LISTMODE:', suffix='%(percent)d%%') as bar:
        for (dirpath, dirnames, filenames) in os.walk(dir_path): 
            dirname = str(Path(dirpath).relative_to(dir_path)) 
            if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname): 
                    new_path = Path(os.path.join(dir_path, dirname))
                    get_LM_date(new_path)
                    with open(new_path/'dump.txt') as f:
                        for line in f.readlines():
                            line_ = line.strip()
                            if line_.startswith('%study date (yyyy:mm:dd):='):
                                d = line_.split(':=')[1]
                                scandate = datetime.strptime(d,'%Y:%m:%d').strftime('%Y%m%d') 
                                name = get_name(str(new_path), regex ='ReconReady') #IF SORTED BY YEAR
                                #name = get_name(str(dirname), regex = '')
                                my_dates[name] = scandate
                                bar.next()
                    os.remove(new_path/'dump.txt')
    return my_dates


#Goes into all CT foldes, writes scan date from each DICOM header into a single file
def check_dates(dir_path):
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
        dirname = str(Path(dirpath).relative_to(dir_path)) 
        if '/IMA' in str(dirname) or '/CT' in str(dirname): 
            new_path = os.path.join(dir_path, dirname) 
            d_name1 = (re.search('(?<=\/)(.*)(\/)', str(dirname))).group(1)
            d_name = get_name(d_name1, regex= "")
            #print(new_path, d_name)
            os.chdir(new_path) 
            os.system(f"for d in * ; do echo strings | dcmdump $d \
                      --search StudyDate| (head -c 30; echo '{d_name}') \
                      >> /homes/michellef/anon.txt; break 1; done")

def find_ct(dir_path):
    check_dates(dir_path)
    anon_patients = {}
    with open('/homes/michellef/anon.txt') as f:
          for line in tqdm(f.readlines(), desc ="Loading CTs"): #progress bar 
              line_ = line.strip()
              name1 = get_name(line_, regex = 'txt')
              #name = get_name(name1, regex='') #IF SORTED BY YEAR
              scan_date = get_name(line_, regex = 'date')
              anon_patients[name1] = scan_date
              sleep(.001)
    os.remove('/homes/michellef/anon.txt')
    return anon_patients


def get_items(my_dict, **tag_):
    my_items = []
    for k,v in my_dict.items():
        if tag_.get('tag') == 'key':
            my_items.append(k)
        if tag_.get('tag') == 'value':
            my_items.append(v)
    return my_items

def concatenation(my_strings,my_numbers):
    result1 = list(map(lambda x, y: [x, y], my_strings, my_numbers))
    #celsius_dict = dict(zip(sorted_ptd.keys(), celsius))
    return result1


#Sort the two dictionaries, compare dates for same patients
def get_info(dir_path):
#    f = open('/homes/michellef/info.txt', 'w')
    dates_ptd = find_ptd(dir_path)
    dates_ct = find_ct(dir_path)

    sorted_ptd = {k: v for k, v in sorted(dates_ptd.items(), key=lambda k: k[0])}
    sorted_ct = {k: v for k, v in sorted(dates_ct.items(), key=lambda k: k[0])}
    #sorted_ptd = dict(sorted(dates_ptd.items(), key=lambda k: k[0])) #CAN ALSO DO THAT??
    #{k: str(v) + '_hooj' for k, v in ... } # STRUCTURE FOR CHANGING DICT OR IMPLEMENTING AN 'IF'
    
    final_dict = {}
    c = 1

    for ptd, ct in zip(sorted_ptd.items(),sorted_ct.items()):
        if ptd[0] == ct[0]:
            final_dict[ptd[0]]= [ptd[1], ct[1]]
            if ptd[1]!=ct[1]:
                print(f'{c}. Mismatch in patient "{ptd[0]}"')
                c+= 1
    #print(f'\n {final_dict}')

    # for k, v in sorted_ptd.items():
    #     print(f'{k} is at {v}')

if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    
    get_info(data_path)

