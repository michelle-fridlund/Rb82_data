# -*- coding: utf-8 -*-
#!/homes/michellef/anaconda3/envs/ claes_test/bin/python
"""
Created on Mon Apr 19 12:49:16 2021

@author: michellef

Call lmparser.py on all files and identify file type
"""

import os
import glob
import argparse
import pydicom
from tqdm import tqdm
from pathlib import Path


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

    if not os.path.exists('header'):
        os.makedirs('header')

    if len(LM) != 0:
        for l in LM:
            name = l.stem

            os.system(f'lmparser.py "{l}" --out_dicom header/"{name}".dcm')


def id_file(pt):
    # Check if header files already exist
    if not (pt/'header').exists():
        parse_lm(pt)

    dicoms = glob.glob("{}/header/*.dcm".format(data_path), recursive=True)

    tags = {}
    for d in dicoms:
        name = d.split('/')[-1].split('.dcm')[0]
        ds = pydicom.dcmread(d)

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
    for k, v in tqdm(tags.items()):
        if 'LISTMODE' in v:
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


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument("--data", "-d",
                               help="Data source directory path with pdt files", required=True)

    parser.add_argument('--dose', dest='dose', type=int,
                        help='percentage of dose to retain (0-100)')

    # Read arguments from the command line
    args = parser.parse_args()
    data_path = Path(args.data)
    dose_level = args.dose

    retain_lm(data_path, dose_level)
