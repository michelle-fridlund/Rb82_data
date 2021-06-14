# -*- coding: utf-8 -*-

import pickle
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict

data_path = '/homes/michellef/my_projects/paediatrics_fdg_data/dicoms'

def find_patients(data_path):
    pts = os.listdir(data_path)
    return np.array(pts)

def train_test(pts):
    pts_train, pts_test = train_test_split(pts, test_size = 0.1)
    return pts_train, pts_test

def write_summary(data_path):
    pts = find_patients(data_path)
    pts_train, pts_test = train_test(pts)

    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(pts_train)
    
    data = defaultdict(list)
    # data = {
    #         'train_%d' % i:[]for i in range(0, kf.get_n_splits(pts_train)-1),
    #         'valid_%d' % i:[]for i in range(0, kf.get_n_splits(pts_valid)-1)
    #         'test': [],
    #         }
    
    for n, (train,valid) in enumerate(kf.split(pts_train)):
        #print(n,len(train),len(valid))
        data['train_%d' % n] = pts_train[train]
        data['valid_%d' % n] = pts_train[valid]
    
    #for p in pts_test:
    data['test'].append(pts_test)
    #print(f"Train: {len(summary['train'])}, Validation: {len(summary['valid'])} patient were found!")
    return data

def build_pickle(data_path):
    output  = str(Path(data_path).parent) 
    os.chdir(output)
    data = write_summary(data_path)
    with open('data.pickle', 'wb') as p:
        pickle.dump(data, p)
    print(pickle.load(open('data.pickle','rb')))

if __name__=="__main__":
    write_summary(data_path)
    build_pickle(data_path)
