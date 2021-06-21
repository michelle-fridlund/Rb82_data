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
from tqdm import tqdm
from pathlib import Path
from rhscripts.utils import LMParser  # claes_test conda env
import seaborn as sns
import matplotlib.pyplot as plt
from shutil import rmtree


# Find all .ptd files
def find_LM(pt):
    ptd = []
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if '.ptd' in f.name:
            ptd.append(f)
    return ptd


def parse_lm(pt):
    LM = find_LM(pt)

    # Crate directory to store header
    if not os.path.exists(f'{pt}/header'):
        os.makedirs(f'{pt}/header')

    if len(LM) != 0:
        for l in LM:
            name = l.stem

            os.system(
                f'lmparser.py "{l}" --out_dicom header/"{name}".dcm')


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


def retain_lm(pt, dose):
    tags = id_file(pt)

    save_dir = '%s/%s_%s' % (str(pt), os.path.basename(str(pt)), dose)

    c = 0
    for k, v in tags.items():
        if 'LISTMODE' in v:
            print(v)
            ptd_path = '%s/%s.ptd' % (str(pt), k)
            c += 1
            if dose is not None:
                os.system(
                    f'lmparser.py "{ptd_path}" --verbose --retain "{dose}" --out_folder "{save_dir}"')
            else:
                print('.', end='', flush=True)
        if 'CALIB' in v:
            print(f'{k} is a CALIBRATION file')

    print(f'{c} LISTMODE files found.')


def find_patients(dir_path, dose):
    c=1
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        dirname = str(Path(dirpath).relative_to(dir_path))
        if '/REST' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname) \
                or '/STRESS' in str(dirname) and 'IMA' not in str(dirname) and 'CT' not in str(dirname):
            new_path = Path(os.path.join(dir_path, dirname))
            save_dir = str((re.search(
                '\/homes\/michellef\/my_projects\/rb82_data\/PET_OCT8_Anonymous_JSReconReady\/(.*)', str(new_path))).group(1))
            print(f'{c}. WE ARE IN {save_dir}')
            # if (new_path/'header').exists():
            #     rmtree(os.path.join(new_path,'header'))
            retain_lm(new_path, dose)
            c+=1

def get_stats(args):
    parser = LMParser(ptd_file=args.ptd_file,  out_folder=args.out_folder,
                      anonymize=args.anonymize, verbose=args.verbose)
    df = parser.return_LM_statistics()
    return df


def plot_prompts(args):
    df = get_stats(args)
    print(df)
    # df.drop(['count'], axis='columns', inplace=True)
    df2 = df[df.type == 'prompt']
    sns.lineplot(x=df.t, y=df.numEvents, hue='type', data=df)
    plt.savefig('/homes/michellef/seaborn-data/prompts_rb82_new_25.png')


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument(
        "--mode", "-m", dest='mode',  help="retain/plot", required=True)
    parser.add_argument("--data", "-d",
                        help="Data source directory path with pdt files")

    parser.add_argument('--dose', dest='dose', type=int,
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
    data_path = Path(args.data)
    dose = args.dose

    # Run on a single patient
    if mode == 'retain':
        retain_lm(data_path, dose)
    # Run on all Rb82 patients
    elif mode == 'parse':
        find_patients(data_path, dose)
    elif mode == 'plot':
        plot_prompts(args)
    else:
        print('No such option.')
