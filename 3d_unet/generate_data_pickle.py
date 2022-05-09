# -*- coding: utf-8 -*-

import argparse
import pickle
import os
from pathlib import Path
import numpy as np
import data_generator as data
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict
#from plotting import find_patients


data_path = ("/homes/michellef/rb82_doses.xlsx")

def find_patients2(data_path):
    pts = find_patients(data_path)
    print(pts)
    return np.array(pts)


""" def find_patients(data_path):
    pts = os.listdir(data_path)
    return np.array(pts) """


def train_test(pts):
    pts_train, pts_test = train_test_split(pts, test_size=0.1)
    return pts_train, pts_test


def write_summary(args, data_path):
    pts = find_patients2(data_path)
    # This function is only used when a test set needs to be created separately
    pts_train_t, pts_test_t = train_test(pts)

    pts_train = []
    pts_test = []

    # Use this argument to append phases later
    if args.extend:
        pts_train = np.array(pts)

    # Regular train/test data split
    else:
        # Important to parse numpy arrays to kf
        pts_train = np.array(pts_train_t)
        pts_test = np.array(pts_test_t)

    kf = KFold(n_splits=6, shuffle=True)
    kf.get_n_splits(pts_train)

    data = defaultdict(list)

    # data = {
    #         'train_%d' % i:[]for i in range(0, kf.get_n_splits(pts_train)-1),
    #         'test_%d' % i:[]for i in range(0, kf.get_n_splits(pts_valid)-1),
    #         'test': [],
    #         }

    for n, (train, valid) in enumerate(kf.split(pts_train)):
        # print(n,len(train),len(valid))
        data['train_%d' % n] = pts_train[train]
        data['test_%d' % n] = pts_train[valid]

    # #for p in pts_test:
    # data['test'].append(pts_test)
    # #print(f"Train: {len(summary['train'])}, Validation: {len(summary['valid'])} patient were found!")
    return data


# This accomodates the rb82 folder structure for pytorch
def sort_rb_phase():
    #data = write_summary(args, data_path)
    data = pickle.load(open('/homes/michellef/my_projects/rhtorch/torch/rubidium2022/data/rb82_final_train_temp.pickle', 'rb'))
    data_rb = {}
    for k, v in data.items():
        # Temporary arrays to store lists
        pts_double = []
        pts_double_suffix = []
        # Double every element in the list of patients
        for pts in v:
            pts_double.extend([pts, pts])

        for n, p in enumerate(pts_double):
            if n % 2 == 0:
                p = p + '_rest'
            else:
                p = p + '_stress'
            pts_double_suffix.append(p)

        data_rb[k] = pts_double_suffix
    with open('/homes/michellef/my_projects/rhtorch/torch/rubidium2022/data/rb82_final_train.pickle', 'wb') as p:
        pickle.dump(data_rb, p)
    return data_rb


def build_pickle(args, data_path):
    #output = str(Path(data_path).parent)
    output = '/homes/michellef/my_projects/rhtorch/torch/rb82/data'
    os.chdir(output)
    print(f'Saved in {output}')

    data = sort_rb_phase(args, data_path) if args.extend \
        else write_summary(args, data_path)

    with open('rb82_6fold_scaled.pickle', 'wb') as p:
        pickle.dump(data, p)
    data = pickle.load(open('rb82_6fold_scaled.pickle', 'rb'))

    t = len(data['train_0'])
    v = len(data['test_0'])
    # tt = len(data['test'][0])
    print(data['test_0'])
    print(f'Train: {t} \n Test: {v}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument(
        "--data", "-d", help="patient directory path")
    required_args.add_argument("--extend", "-e", help="extend for rest/stress suffix?",
                               required=True, type=data.ParseBoolean)

    # Read arguments from the command line
    args = parser.parse_args()
    #data_path = str(args.data)

    #build_pickle(args, data_path)

    sort_rb_phase()
