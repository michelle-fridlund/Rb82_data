"""
Created on Thu Mar 24 14:37 2022

##############################################################################
Script for preprocessing batches of rb82 raw data
##############################################################################

@author: michellef
"""

import os
import argparse
from shutil import rmtree
from pathlib import Path 
from identify_LM_date import write_pickle, sort_patients


def remove_patients(data_path, pkl_name):
    # Print incorrect patient names
    tracer, dose = sort_patients(data_path, pkl_name)

    num_p = len(tracer) + len(dose)

    if args.delete:

        for t in tracer:
            rmtree(os.path.join(data_path, t))
        for d in dose:
            rmtree(os.path.join(data_path, d))

        print(f'Deleted {num_p} patients...')

if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d", help="Data source directory path", required=True)
    parser.add_argument('--delete', dest='delete', type=bool, default=False, help='delete wrong patient types: True/False')
    required_args.add_argument("--pkl", "-p", help=".pickle file name", required=True)

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    pkl_name = str(args.pkl)
    
    remove_patients(data_path, pkl_name)