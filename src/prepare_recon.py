"""
Created on Thu Mar 24 14:37 2022

##############################################################################
Script for preprocessing batches of rb82 raw data
##############################################################################

@author: michellef
"""
import re
import os
import argparse
from shutil import rmtree
from pathlib import Path
from identify_LM_date import sort_patients


# Find patients by tag
def find_patients(dir_path):
    c = 0
    ct_paths = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path): 
        dirname = str(Path(dirpath).relative_to(dir_path)) 
        if 'ACCT/' in str(dirname): 
            new_path = os.path.join(dir_path, dirname)
            ct_paths.append(new_path)
            c+=1
    print(f'{c} moved.')
    return ct_paths


def move_patients(dir_path):
    ct_paths = find_patients(dir_path)

    for ct in ct_paths:
        name = (re.search('(\/homes\/michellef\/my_projects\/rb82_data\/rb82_Mar23)\/(?<=\/)(.*)', ct)).group(2)
        dest = os.path.join('/homes/claes/data_shared/michelle', name)
        dest = str(dest)
        ct = str(ct)
        #print(dest)
        if not os.path.exists(dest):
            os.makedirs(dest)

        os.system(f"mv '{ct}' '{dest}'")


def remove_patients(data_path, pkl_name):
    # Print incorrect patient names
    tracer, dose = sort_patients(data_path, pkl_name)

    num_p = len(tracer) + len(dose)

    if args.delete:

        for t in tracer:
            if os.path.exists(os.path.join(data_path, t)):
                rmtree(os.path.join(data_path, t))
        for d in dose:
            if os.path.exists(os.path.join(data_path, d)):
                rmtree(os.path.join(data_path, d))

        print(f'Deleted {num_p} patients...')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)
    parser.add_argument('--delete', dest='delete', type=bool, default=False, help='delete wrong patient types: True/False')
    required_args.add_argument("--pkl", "-p", help=".pickle file name")

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    pkl_name = str(args.pkl)
    
    remove_patients(data_path, pkl_name)
