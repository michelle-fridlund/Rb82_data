# -*- coding: utf-8 -*-

import argparse 
import pickle
import os
from pathlib import Path
import numpy as np
import data_generator as data
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict


def find_patients(data_path):
    pts = os.listdir(data_path)
    return np.array(pts)


def train_test(pts):
    pts_train, pts_test = train_test_split(pts, test_size=0.1)
    return pts_train, pts_test


def write_summary(args, data_path):
    pts = find_patients(data_path)
    pts_train_t, pts_test_t = train_test(pts)

    pts_train = []
    pts_test = []
    print(pts_test_t)
    if args.extend:
        pts_train_double = []
        
        for i in pts:
            pts_double.extend([i, i])

        for n,p in enumerate(pts_double):
            if n%2==0:
                p = p + '_rest'
            else:
                p = p + '_stress'
            pts_train.append(p)
        pts_train = np.array(pts_train)

        print(len(pts_train))
            
    else:
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

    # for n, (train,valid) in enumerate(kf.split(pts_train)):
    #     # print(n,len(train),len(valid))
    #     data['train_%d' % n] = pts_train[train]
    #     data['test_%d' % n] = pts_train[valid]

    # #for p in pts_test:
    # data['test'].append(pts_test)
    # #print(f"Train: {len(summary['train'])}, Validation: {len(summary['valid'])} patient were found!")
    return data


def build_pickle(args, data_path):
    output = str(Path(data_path).parent)
    os.chdir(output)
    print(f'Saved in {output}')
    data = write_summary(args, data_path)
    with open('rb82_6fold_sorted.pickle', 'wb') as p:
        pickle.dump(data, p)
    data = pickle.load(open('rb82_6fold_sorted.pickle','rb'))
    print(data)
    # t = len(data['train_0'])
    # v = len(data['valid_0'])
    # tt = len(data['test'][0])
    # print(f'Train: {t} \n Val: {v}, Test: {tt}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    # Add long and short argument
    required_args.add_argument(
        "--data", "-d", help="patient directory path", required=True)
    required_args.add_argument("--extend", "-e", help="extend for rest/stress suffix?",
                               required=True, type=data.ParseBoolean, default=False)


    # Read arguments from the command line
    args = parser.parse_args()
    data_path = str(args.data)

    write_summary(args, data_path)
    build_pickle(args, data_path)
