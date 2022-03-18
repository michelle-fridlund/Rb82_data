# -*- coding: utf-8 -*-
#!/homes/michellef/anaconda3/envs/claes_test/bin/python
"""
Created on Mon Apr 19 12:49:16 2021

@author: michellef

Call lmparser.py on all files and identify file type
"""

import os
import re
import glob
import argparse
import pydicom
import pickle
from tqdm import tqdm
from pathlib import Path
from rhscripts.utils import LMParser  # claes_test conda env
import seaborn as sns
import matplotlib.pyplot as plt
from shutil import rmtree
import pandas as pd


# Find all .ptd files
def find_LM(pt):
    ptd = []
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if '.ptd' in f.name:
            ptd.append(f)
    return ptd


# Create dicoms headers for .ptd files
def parse_lm(pt):
    LM = find_LM(pt)

    # Crate directory to store header
    if not os.path.exists(f'{pt}/header'):
        os.makedirs(f'{pt}/header')

    if len(LM) != 0:
        for l in LM:
            name = l.stem

            os.system(
                f'lmparser.py --ptd_file "{l}" --out_dicom header/"{name}".dcm')


# Identify .ptd file type from generated header
def id_file(pt):
    # Check if header files already exist
    if not (pt/'header').exists():
        parse_lm(pt)

    dicoms = glob.glob("{}/header/*.dcm".format(pt), recursive=True)

    tags = {}
    for d in dicoms:
        # Get filename
        name = d.split('/')[-1].split('.dcm')[0]
        ds = pydicom.dcmread(d)
        # Extract LM file type from DICOM header
        try:
            tag = str(ds[0x29, 0x1008].value)
            tags[name] = tag
        except:
            print('No tag found')

    return tags


# Initiate lmparser.py from utils with input dose
def retain_lm(pt, dose):
    tags = id_file(pt)

    # Create save directory in the parent folder
    save_dir = '%s/%s_%s' % (str(pt), os.path.basename(str(pt)), dose)

    c = 0
    for k, v in tags.items():
        if 'LISTMODE' in v:
            print(v)
            ptd_path = '%s/%s.ptd' % (str(pt), k)
            c += 1
            print(ptd_path)
            if dose is not None:
                os.system(
                    f'lmparser.py --ptd_file "{ptd_path}" --verbose --fake_retain "{dose}"') # only update the header
                    #f'lmparser.py --ptd_file "{ptd_path}" --verbose --retain "{dose}" --out_folder "{save_dir}"')
            else:
                print('.', end='', flush=True)
        if 'CALIB' in v:
            print(f'{k} is a CALIBRATION file')

    print(f'{c} LISTMODE files found.')


# Find all available patient LISTMODE fileE paths and call lmparser on them
def find_patients_fdg(data_path, dose):
    c = 1
    patients = os.listdir(data_path)
    for p in patients:
        new_path = Path(os.path.join(data_path, p))
        retain_lm(new_path, dose)
        print(f"{c}. {new_path}")
        c += 1


# Same as above, but accomodates Rb82 folder structure
def find_patients_rb(dir_path, dose):
    c = 1
    # Only need the following directories
    directories = ['2016_25p', '2017_25p', '2018_25p', '2019_25p', '2020_25p']
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
            and dirname.split('/')[0] in directories \
            or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
            and dirname.split('/')[0] in directories:
            new_path = Path(os.path.join(dir_path, dirname))
            #save_dir = str((re.search(
            #    '\/homes\/michellef\/my_projects\/rb82_data\/PET_LMChopper_OCT8/(.*)', str(new_path))).group(1))
            #print(f'{c}. WE ARE IN {save_dir}')
            #if (new_path/'header').exists():
            #     rmtree(os.path.join(new_path,'header'))
            retain_lm(new_path, dose)
            c += 1


# Get statistics using inheritance
def get_stats(args):
    parser = LMParser(ptd_file=args.ptd_file,  out_folder=args.out_folder,
                      anonymize=args.anonymize, verbose=args.verbose)
    df = parser.return_LM_statistics()
    return df


# Plot from stats (hard-coded)
def plot_prompts(args):
    df = get_stats(args)
    # print(df)
    # df.drop(['count'], axis='columns', inplace=True)
    #df2 = df[df.type == 'events']
    df.to_pickle(os.path.join(str(args.data), '22.pkl'))
    # df = pd.read_pickle(os.path.join(str(args.data),'phantom_50.pkl'))
    sns.lineplot(x=df.t, y=df.numEvents, data=df, hue='type')
    plt.savefig('/homes/michellef/seaborn-data/23aug_22.png')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument(
        "--mode", "-m", dest='mode',  help="retain/parse/plot", type=str, required=True)
    required_args.add_argument(
        "--tracer", dest='tracer', help="fdg/rb", type=str, required=True)

    parser.add_argument("--data", "-d",
                        help="Data source directory path with pdt files")

    parser.add_argument('--dose', dest='dose', type=float,
                        help='percentage of dose to retain (0-100)')
    # Filename only
    parser.add_argument("--ptd_file", help='Input PTD LLM file', type=str)
    parser.add_argument(
        "--retain", help='Percent (float) of LMM events to retain (0-100)', type=float)
    parser.add_argument(
        "--out_folder", help='Output folder for chopped PTD LLM file(s)', type=str)
    parser.add_argument(
        "--out_filename", help='Output filename for chopped PTD LLM file', type=str)
    parser.add_argument(
        "--seed", help='Seed value for random', default=11, type=int)
    parser.add_argument(
        "--out_dicom", help='Save DICOM header to file', type=str)
    parser.add_argument('--anonymize', action='store_true')
    parser.add_argument("-v", "--verbose",
                        help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    # Read arguments from the command line
    mode = str(args.mode)
    tracer = str(args.tracer).lower()
    data_path = Path(args.data)
    dose = args.dose

    # Run on a single patient
    if mode == 'retain':
        retain_lm(data_path, dose)
    # Run on all patients
    elif mode == 'parse':
        if 'fdg' in tracer:
            find_patients_fdg(data_path, dose)
        elif 'rb' in tracer:
            find_patients_rb(data_path, dose)
        else:
            print('Wrong tracer.')
    elif mode == 'plot':
        plot_prompts(args)
    else:
        print('No such option.')
