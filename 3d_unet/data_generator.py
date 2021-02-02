"""
Jan 18 15:56

michellef
##############################################################################
Upload 3D PET data
##############################################################################
"""

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import os
import glob
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from random import shuffle
import threading
import pickle 
import DataAugmentation3D


class DCMDataLoader(object):
    
    def __init__(self, args, mode):
        #paths to dicoms files
        self.data_path = args.data_path
        self.ld_path = args.ld_path
        self.hd_path = args.hd_path
        #self.state_name = args.state_name
        
        #pickle file
        self.summary = pickle.load(open('%s/data.pickle' % self.data_path, 'rb'))
        self.train_or_test = args.train_or_test
        self.fold = 0
        self.mode = '{}_{}'.format(self.train_or_test, self.fold)
        #image params
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        #training params
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size
        
        #data augmentation
        self.augment = args.augment 
        self.augmentation_params = {'rotation range': [5, 5 ,5],
                                    'shift_range': [0.05 ,0.05 , 0.05],
                                    'shear_range': [2, 2, 0],
                                    'zoom_lower' : [0.9, 0.9, 0.9],
                                    'zoom_upper' : [1.2, 1.2, 1.2],
                                    'zoom_independent' : True,
                                    'flip_axis' : [1, 2],          
                                    #'X_shift_voxel' :[2, 2, 0],
                                    #'X_add_noise' : 0.1,
                                    'fill_mode' : 'reflect'
                                   }
        self.augment3d = DataAugmentation3D(**self.augmentation_params) 
        
        self.n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary['train_0'])
        self.n_batches /= self.batch_size
        
        
    #Transform DICOMS into numpy
    def load_nifti(self, path):
        nifti = [nib.load(i) for i in glob.glob("{}/*.nii.gz".format(path), recursive = True)]
        return nifti
    #Return stack of all slices
    def nifti2numpy(self, nifti):
        pixels = [nifti.get_fdata() for n in nifti]
        return np.array(pixels, dtype = np.float16)
    
    def augment_data(self):
        X = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size) + (self.input_channels,))
        y = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size) + (self.output_channels,))

        for i in range(self.batch_size):
            ld_, hd_ = self.load_data(self.summary, 'train_0')
            X[i,...] = self.ld_
            y[i,...] = self.hd_.reshape((self.image_size, self.image_size, self.patch_size) + (self.output_channels,))

            X, y = self.augment3D.random_transform_batch(X,y)

        return X, y

       
    def load_data(self, sumary, mode, z = None):
        patients = self.summary[mode]
        
    #Load and reshape all patient data
        for patient in patients:
        
            ld_path = '%s/%s/%s' % (self.data_path, self.ld_path, patient)
            hd_path = '%s/%s/%s' % (self.data_path, self.hd_path, patient)
            
            ld_data = self.nifty2numpy(self.load_nifti(ld_path))
            hd_data = self.nifty2numpy(self.load_nifti(hd_path))
            
            #print(f'Load complete: {len(ld_data)} LD and {len(hd_data)} HD dcm found for patient {patient}')
            
            self.ld_ = ld_data.reshape(128,128,111,1)
            self.hd_ = hd_data.reshape(128,128,111,1)
            
            if mode == 'train' and self.augment:
                self.ld_, self.hd_ = self.augment3D.random_transform_batch(self.ld_,self.hd_)
            
            # --- Determine slice
            if z == None:
                z = np.random.randint(8,111-8,1)[0]
        
                ld_stack = self.ld_[:,:,z-8:z+8,:]
                hd_stack = self.hd_[:,:,z-8:z+8,:]

            return ld_stack, hd_stack
            #print(f'{ld_.shape} {hd_.shape} {self.batch_size}')



#main.py parsers 
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError ('Cannot parse string into boolean.')

def Capitalise(s):
    return s.upper()