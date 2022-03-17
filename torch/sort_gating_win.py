#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 2022 
@author: michellef
"""

import os
import csv

phase = 'STRESS'

#Paths to original and temporary .csv files
source = 'C:\\Users\\pet\\Desktop\\michelle\\2019_test'
filename1 = f'{phase}-Converted\{phase}-CG-00\{phase}-CG-00.csv'
filename2 = f'{phase}-Converted\{phase}-CG-00\{phase}-CG-00_edit.csv'

patients = os.listdir(source)


# Create full paths for all patients 
def find_patients(patients):
    csv_paths = []
    temp_paths = []
    for patient in patients:
        print(patient)
        file1 = os.path.join(source, patient, filename1)
        file2 = os.path.join(source, patient, filename2)
        csv_paths.append(file1)
        temp_paths.append(file2)
    return csv_paths, temp_paths
    

# Read original .csv and return necessary lines as list
def read_csv(source, dest):
    with open(source, 'r') as inp, open(dest, 'w') as out:
        writer = csv.writer(out)
        gates = []
        for row in csv.reader(inp):
            if row[0] == 'Time(ms)' or int(row[0]) >= 150000 and int(row[0]) <= 360000:
                writer.writerow(row)
                gates.append(row)
    return gates


# Write to temp .csv and replace the original
def write_csv(source, dest):
    gates = read_csv(source, dest)
    if len(gates) != 0:
        print(f'{len(gates)} lines written.')
        os.remove(source)
        os.rename(dest, source)


def prep_recon(patients):
    csv_paths, temp_paths = find_patients(patients)
    for csv, temp in zip(csv_paths, temp_paths):
        write_csv(csv, temp)
    print(f'{len(csv_paths)} patient done!')
    
prep_recon(patients)


