#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:05:11 2020

@author: michellef
"""
import pickle
from datetime import datetime
from pathlib import Path
import glob
import os
import re

import pprint


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020

@author: michellef
"""
def update_pickle(patients):
    # p = pickle.load(open('patient.pkl','rb'))
    # pickle.dump(p,open('patient.pkl','wb'))
    return


def find_files(dir_path):
    directories = {}
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if dirpath == str(dir_path) or '/' in dirname:
            continue
        directories[dirname] = [os.path.basename(x) for x in glob.glob("{}/*.ptd".format(dirpath), recursive = True)]

    return directories


# def find_files_alternative(dir_path):
#     directories = {}
#     if not dir_path.is_dir():
#         return None

#     for cur_path in dir_path.iterdir():
#         if not cur_path.is_dir():
#           continue

#         dirname = str(cur_path.relative_to(dir_path))
#         if not directories.get(dirname):
#             directories[dirname] = []

#         for inner_path in cur_path.iterdir():
#             if not inner_path.is_file():
#                 continue

#             filename = str(inner_path.relative_to(cur_path))
#             directories[dirname].append(filename)

#     pprint.pprint(directories)
#     return directories

def id_files(dir_path):
    dirlist = find_files(dir_path)
    patients = {}
    occurencies = {}
    for key, filename in dirlist.items():
        patient = {
            'LISTMODE': [],
            'PHYSIO': [],
            'CALIBRATION': [],
            'OTHER': [],
        }

        patient_name = re.search('^[^0-9]*', key).group()
        if occurencies.get(patient_name):
            occurencies[patient_name] = (occurencies[patient_name][0], occurencies[patient_name][1] + 1)
        else:
            occurencies[patient_name] = (key, 1)

        # print(patient_name, occurencies[patient_name])
        if occurencies[patient_name][1] > 1:
            patients.pop(occurencies[patient_name][0], None)
            continue

        for item in filename:
            if 'LISTMODE' in item or '.LM.' in item:
                patient['LISTMODE'].append(item)
            elif 'PHYSIO' in item:
                patient['PHYSIO'].append(item)
            elif 'CALIBRATION' in item:
                patient['CALIBRATION'].append(item)
            else:
                patient['OTHER'].append(item)

        if len(patient['LISTMODE']) != 2 or len(patient['PHYSIO']) != 2 or len(patient['CALIBRATION']) != 1:
            continue

        patients[key] = patient

    return patients


def print_pt():
    p = pickle.load(open('patient.pkl','rb'))
    print(p.keys(), p.items())


if __name__ == "__main__":

    dir_path = Path('/home/quantumcoke/Rb82/data')
    patients = id_files(dir_path)
    pprint.pprint(patients)
    #update_pickle(patients)
    #print_pt()


