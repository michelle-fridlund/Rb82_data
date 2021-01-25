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
import pydicom
from random import shuffle
import threading
import pickle 
import DataAugmentation3D as augment3d


class DCMDataLoader(object):
    
    def __init__(self, args, mode):
        #paths to dicoms files
        self.data_path = args.data_path
        self.ld_path = args.ld_path
        self.hd_path = args.hd_path
        self.state_name = args.state_name
        
        #pickle file
        self.summary = pickle.load(open('%s/data.pickle' % self.data_path, 'rb'))
        #image params
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        #training params
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size
        
        self.augment = args.augment 
        
        self.n_batches = len(self.summary['train']) 
        self.n_batches /= self.batch_size
        

    #Transform DICOMS into numpy
    def load_dicom(self, path):
        slices = [pydicom.read_file(i) for i in glob.glob("{}/*.ima".format(path), recursive = True)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    #Return stack of all slices
    def dcm2numpy(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image, dtype = np.float16)
       
    def load_data(self, sumary, mode, z = None):
        patients = self.summary[mode]
        
    #Load and reshape all patient data
        for patient in patients:
        
            ld_path = '%s/%s/%s/%s' % (self.data_path, self.ld_path, patient, self.state_name)
            hd_path = '%s/%s/%s/%s' % (self.data_path, self.hd_path, patient, self.state_name)
            
            ld_data = self.dcm2numpy(self.load_dicom(ld_path))
            hd_data = self.dcm2numpy(self.load_dicom(hd_path))
            
            #print(f'Load complete: {len(ld_data)} LD and {len(hd_data)} HD dcm found for patient {patient}')
            
            ld_ = ld_data.reshape(128,128,111,1)
            hd_ = hd_data.reshape(128,128,111,1)
            
            # --- Determine slice
            if z == None:
                z = np.random.randint(8,111-8,1)[0]
        
                ld_stack = ld_[:,:,z-8:z+8,:]
                hd_stack = hd_[:,:,z-8:z+8,:]

            return ld_stack, hd_stack
            #print(f'{ld_.shape} {hd_.shape} {self.batch_size}')

    def load_train_data(self, args, mode):
        X = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size) + (self.input_channels,))
        y = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size) + (self.output_channels,))

        for i in range(self.batch_size):
            ld_, hd_ = self.load_data(self.summary, mode)
            X[i,...] = ld_
            y[i,...] = hd_.reshape((self.image_size, self.image_size, self.patch_size) + (self.output_channels,))
                
        if mode == 'train' and self.augment == True:
            X, y = self.augment3D.random_transform_batch(X,y) #TODO - See how this actually works...

        return X, y


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