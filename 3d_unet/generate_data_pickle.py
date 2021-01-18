# -*- coding: utf-8 -*-

import pickle
import os

"""

Save pickle file of train and validation pts
Should have indexes "train" and "test" or "train_X" and "valid_X" 
where X is integer from 0, representing the LOG in k-fold.

"""

summary = { 'train': [], 'valid': [] }

pts = os.listdir('/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc')

summary['train'].append(pts[0])
summary['train'].append(pts[1])
summary['valid'].append(pts[10])
summary['valid'].append(pts[11])

with open('test_dat.pickle', 'wb') as file_pi:
    pickle.dump(summary,file_pi)
    
    
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