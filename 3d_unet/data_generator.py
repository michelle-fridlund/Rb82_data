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
#import pydicom
from random import shuffle
import threading
import pickle 
import DataAugmentation3D as augment3d


class DCMDataLoader(object):
    def __init__(self, args):
       
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
        #self.depth = depth #stack of slices??

        #training params
        #self.batch_size = args.batch_size
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.augment = args.augment 
        
        self.batch_size = len(self.summary['train']) 
        
        #return self.load_train_data(self.summary)

    #Transform DICOMS into numpy
    def load_dicom(self, path):
        slices = [pydicom.read_file(i) for i in glob.glob("{}/*.ima".format(path), recursive = True)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    
    def dcm2numpy(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image, dtype = np.float16)
       
    def load_data(self, mode):
        patients = self.summary[mode]
    #Load an reshape all patient data
        for patient in patients:
        
            ld_path = '%s/%s/%s/%s' % (self.data_path, self.ld_path, patient, self.state_name)
            hd_path = '%s/%s/%s/%s' % (self.data_path, self.hd_path, patient, self.state_name)
            
            ld_data = self.dcm2numpy(self.load_dicom(ld_path))
            hd_data = self.dcm2numpy(self.load_dicom(hd_path))
            
            print(f'Load complete: {len(ld_data)} LD and {len(hd_data)} HD dcm found for patient {patient}')
            
            ld_ = np.memmap(ld_data, dtype='double', mode='r')
            ld_ = ld_.reshape(128,128,-1,2)
            hd_ = np.memmap(hd_data, dtype='double', mode='r')
            hd_ = hd_.reshape(128,128,-1)

        return ld_, hd_

    def load_train_data(self, args, mode):
        X = np.empty((self.batch_size,) + self.img_size + (self.input_channels,))
        y = np.empty((self.batch_size,) + self.img_size + (self.output_channels))

        for i in range(self.batch_size):
            ld_, hd_ = self.load_data(mode)
    
            X[i,...] = ld_
            y[i,...] = hd_.reshape(self.img_res + (self.output_channels))
                           
            if mode == 'train' and self.augment:
                X, y = self.augment3D.random_transform_batch(X,y)
                return X, y
            else:
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