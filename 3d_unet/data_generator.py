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
from glob import glob
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import dicom
#import pydicom
from random import shuffle
import threading
import pickle 
import DataAugmentation3D as augement3d

hd_name = '100p_STAT'
ld_name = '50p_STAT'
state_name = 'REST'
root = '/homes/michellef/Rb82/data/Dicoms_OCT8'

class DCMDataLoader(object):
    def __init__(self, data_path, ld_path, hd_path, \
                 image_size = 128, patch_size = 64, depth = 1, \
                 batch_size = 1, model = 'unet', input_channels = 2, \
                 output_channels = 1, augment = False):
       
        #paths to dicoms files
        self.data_path = data_path
        self.ld_path = ld_path
        self.hd_path = hd_path
        
        #pickle file
        self.summary = pickle.load(open('%s/data.pickle' % self.data_path, 'rb'))
        #image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth

        #training params
        self.batch_size = batch_size
        self.model = model
        self.output_channels = output_channels
        self.augment = augment
        
        self.n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary['train_0'])
        self.n_batches /= self.batch_size


# Transform DICOMS into num
    def load_dicom(self, path):
        slices = [dicom.read_file(i) for i in glob.glob("{}/*.ima".format(path), recursive = True)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    
    def dcm2numpy(self, slices):
        #slices = load_dicom(path)
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image, dtype = np.float32)
       
    def load_data(self, mode):
        # summary_file = '%s/data.pickle' % self.data_path
        # summary = pickle.load(open(summary_file, 'rb'))
        indices = np.random.randint(0, len(self.summary[mode]), 1)
        patient = self.summary[mode][indices[0]]
    
        ld_path = '%s/%s/%s/%s' % (self.data_path, self.ld_name, patient, self.state_name)
        hd_path = '%s/%s/%s/%s' % (self.data_path, self.hd_name, patient, self.state_name)
            
        ld_data = self.dcm2numpy(self.load_dicom(ld_path))
        hd_data = self.dcm2numpy(self.load_dicom(hd_path))
        
        print(f'Load complete: {len(ld_data)} LD and {len(hd_data)} HD patients found')
        
        #if augment:
            #call augment3d
        return ld_data, hd_data
        


