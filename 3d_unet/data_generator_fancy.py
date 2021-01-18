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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import dicom
import math
#import pydicom
from random import shuffle
import threading
import pickle 


class DataGenerator():

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.img_res = config['input_patch_shape']
        self.input_channels = config['input_channels']
        self.output_channels = config['output_channels']
        self.augmentation = config['augmentation']
        self.augmentation_params = config['augmentation_params']
        #   self.data_augmentation = DataAugmentation3D(**self.augmentation_params)

        self.summary = pickle.load( open(config['data_pickle'], 'rb') )
        self.data_folder = config['data_folder']
        
        self.state_name = 'REST'
        
        self.dat_name = '50p_STAT'
        self.tgt_name = '100p_STAT'

        self.n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary['train_0'])
        self.n_batches /= self.batch_size
        
    def load_dicom(self, summary, state_name):
        indices = np.random.randint(0, len(summary[mode]), 1)
        patient = summary[mode][indices[0]]

        path = '%s/%s/%s' % (root, patient, state_name)
    
        slices = [dicom.read_file(i) for i in glob.glob("{}/*.ima".format(path), recursive = True)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    
    def dcm2numpy(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image, dtype = np.float32)
    

    def generate(self, train_or_test):
        while 1:
            X, y = self.__data_generation(train_or_test)
            yield X, y

    def __data_generation(self, train_or_test):
        X = np.empty( (self.batch_size,) + self.img_res + (self.input_channels,) )
        y = np.empty( (self.batch_size,) + self.img_res + (self.output_channels,) )

        for i in range(self.batch_size):

            dat,tgt = self.load(train_or_test,load_mode='memmap')

            X[i,...] = dat
            y[i,...] = tgt.reshape(self.img_res + (self.output_channels,))
            
        if train_or_test.startswith('train') and self.augmentation:
            X, y = self.data_augmentation.random_transform_batch(X,y)

        return X,y

    def load(self, mode, z=None, return_studyid=False, load_mode='npy'):

        indices = np.random.randint(0, len(self.summary[mode]), 1)
        stats = self.summary[mode][indices[0]]

        # --- Load data and labels 
        fname_dat = '%s/%s/%s' % (self.data_folder, stats, self.dat_name)
        fname_tgt = '%s/%s/%s' % (self.data_folder, stats, self.tgt_name)

        if load_mode == 'npy':
            dat = np.load(fname_dat)
            tgt = np.load(fname_tgt)
        elif load_mode == 'memmap':
            dat = np.memmap(fname_dat, dtype='double', mode='r')
            dat = dat.reshape(128,128,-1,2)
            tgt = np.memmap(fname_tgt, dtype='double', mode='r')
            tgt = tgt.reshape(128,128,-1)

        # --- Determine slice
        if z == None:
            z = np.random.randint(8,111-8,1)[0]
        
        dat_stack = dat[:,:,z-8:z+8,:]
        tgt_stack = tgt[:,:,z-8:z+8]

        if return_studyid:
            return dat_stack, tgt_stack, stats
        else:
            return dat_stack, tgt_stack
        
def set_root(loc=None):
    global root, summary

    if loc is None:
        paths = ['PETCT']
        for path in paths:
            if os.path.exists(path):
                loc = path
                break

    #root = loc
    root = '/homes/michellef/Rb82/data/Dicoms_OCT8'
    summary_file = '%s/data_suv.pickle' % root
    summary = pickle.load(open(summary_file, 'rb'))

# --- Set root and summary
global root, summary
set_root()