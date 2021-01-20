# -*- coding: utf-8 -*-

import pickle
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

data_path = '/homes/michellef/my_projects/Rb82/data/Dicoms_OCT8/100p_STAT'

def find_patients(data_path):
    pts = os.listdir(data_path)
    pts_train, pts_val = train_test_split(pts, test_size = 0.25)
    return pts_train, pts_val

def write_summary(data_path):
    summary = {'train': [], 'valid': []}
    pts_train, pts_val = find_patients(data_path)
    for t in pts_train:
        summary['train'].append(t)
    for v in pts_val:
        summary['valid'].append(v)
    print(f"Train: {len(summary['train'])}, Validation: {len(summary['valid'])} patient were found!")
    return summary

def build_pickle(data_path):
    output  = str(Path(data_path).parent) 
    os.chdir(output)
    summary = write_summary(data_path)
    # with open('data.pickle', 'wb') as p:
    #     pickle.dump(summary, p)
    print(pickle.load(open('data.pickle','rb')))

    
if __name__=="__main__":
    
    build_pickle(data_path)
    
""" 

#############################################################

Example below show code for splitting 6 fold cross validation
Note: No exsplicit care is taken so that double scan of patients are not both in train and validation.
      If that is the case for your project - this has to be handled !!

#############################################################

datafolder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'

patients = [f for f in os.listdir(datafolder) if os.path.exists(os.path.join(datafolder,f,'FDG_100_SUV.mnc'))]
patients = np.array(patients)

from sklearn.model_selection import KFold
import numpy as np
import os
import pickle

kf = KFold(n_splits=6,shuffle=True)
kf.get_n_splits(patients)

data = {}
for G, (train,test) in enumerate(kf.split(patients)):
    print(G,len(train),len(test))
    data['train_%d' % G] = patients[train]
    data['valid_%d' % G] = patients[test]
    
pickle.dump( data, open('data_6fold.pickle','wb') )
"""