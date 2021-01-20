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
import DataAugmentation3D as augement3d


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
        self.batch_size = args.batch_size
        #self.output_channels = output_channels
        self.augment = args.augment 
        
        self.n_batches = len(self.summary['train']) 
        self.n_batches = self.batch_size
        
        return self.load_train_data(self.summary)


    # Transform DICOMS into numpy
    def load_dicom(self, path):
        slices = [pydicom.read_file(i) for i in glob.glob("{}/*.ima".format(path), recursive = True)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    
    def dcm2numpy(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        return np.array(image, dtype = np.float32)
       
    def load_train_data(self, summary):
        patients = self.summary['train']
        
        for patient in patients:
        
            ld_path = '%s/%s/%s/%s' % (self.data_path, self.ld_path, patient, self.state_name)
            hd_path = '%s/%s/%s/%s' % (self.data_path, self.hd_path, patient, self.state_name)
            
            ld_data = self.dcm2numpy(self.load_dicom(ld_path))
            hd_data = self.dcm2numpy(self.load_dicom(hd_path))
            
            print(f'Load complete: {len(ld_data)} LD and {len(hd_data)} HD dcm found for patient {patient}')
            
            # if augment:
            #     call augment3d
                
            #return ld_data, hd_data
            print(ld_data, hd_data)
            

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