"""
Created on Wed Jun 30 14:55:22 2021

Preprocess data for torch.io
"""
import os
import glob
import pickle
import numpy as np
import nibabel as nib 
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler


class Data_Preprocess(object):
    def __init__(self, args, hd_name = 'pet_100p_ekg_gate8', ld_name= 'pet_25p_ekg_gate8',
                 ct_name = 'mask', extension = '.nii.gz'):
        # PET norm value from arguments
        # 232429.9 (25, 99.8)
        self.norm = args.norm 

        # Scale factor for different dose levels
        self.scale = args.scale
        
        self.hd_name = hd_name
        self.ld_name = ld_name
        self.ct_name = ct_name
        self.extension = extension
        
        # Paths to original files
        self.data_path = args.data_path
        # List of patients in data dir

        #Test group
        pk = pickle.load(open('/homes/michellef/my_projects/rhtorch/torch/rb82/data/rb82_6fold_sorted.pickle', 'rb'))
        #self.summary = pk['test_0']

        self.summary = os.listdir(self.data_path)
        
    def load_nifti(self, file):
        return nib.load(file)
    
    
    # Transform DICOMS into numpy
    def nifti2numpy(self, nifti):
        try:
            d_type = nifti.header.get_data_dtype() #Extract data type from nifti header
            return np.array(nifti.get_fdata(), dtype=np.dtype(d_type))
        except:
            return None
        

    # Scale low-dose input
    def scale_pet_dose(self, pixels):
        return pixels*self.scale  # 1/4 factor for 25% dose 
        

    # PET hard normalisation
    def normalise_pet(self, pixels):
        return pixels/self.norm
    
    
    def normalise_robust(self, pixels):
        scaler = RobustScaler()
        print(f'Before: {np.mean(pixels)}')
        # Need to flatten array when passing to RobustScaler
        pixels = scaler.fit_transform(pixels.reshape(-1, pixels.shape[-1])).reshape(pixels.shape)
        print(f'After: {np.mean(pixels)}')
        return pixels

    
    def normalise_ct(self, pixels):
        #return (pixels + 1024.0) / 4095.0
        return 1.0 if pixels.any()==2 else 0.0


    # Normalise binary mask
    def normalise_mask(self, pixels):
        mask = np.where(pixels == 2)
        no_mask = np.where(pixels != 2)

        pixels[mask] = 1.0
        pixels[no_mask] = 0.0

        return np.array(pixels) 


    # Choose normalisation based on input type
    def prep_numpy(self, numpy, **mode):
        if mode.get("mode") == "hd":
            return self.normalise_pet(numpy)
        elif mode.get("mode") == "ld":
            return self.scale_pet_dose(self.normalise_pet(numpy))
        elif mode.get("mode") == "ct":
            # Some CTs have an extra slice at the end
            if numpy.shape[2] == 112:
                return self.normalise_mask(numpy)[:,:,0:111] 
            elif numpy.shape[2] == 111:
                return self.normalise_mask(numpy) 
            else:
                print('Oddly-sized numpy array')
        else:
            print('Invalid input type: hd/ld/ct')
            
            
    # Concatenate load/save path strings
    def create_paths(self, patient, filename):
        load_path = '%s/%s/%s%s' % (self.data_path, patient, filename, self.extension)
        save_path = '%s/%s/%s%s%s' % (self.data_path, patient, filename, '_norm', self.extension)
        return [load_path, save_path]

            
    # Create normalised PET nifti
    def save_nifti(self, nifti, numpy, save_path):
        image = nib.Nifti1Image(numpy, nifti.affine, nifti.header)
        nib.save(image, save_path)
        
        
    # Create normalised CT nifti
    def transform_nifti(self, nifti, nifti_pet, numpy, save_path):
        # Need to downsample CT from 512x512 to 128x128
        numpy = numpy[::4,::4,]
        new_header = nifti.header.copy()
        # Use affine matrix to match PET input
        xform = nifti_pet.affine
        img = nib.Nifti1Image(numpy, xform, header = new_header)
        nib.save(img, save_path)
        print('Transforming CT...')
        
        
    def create_new_nifti(self, load_path, load_path2, save_path, mode):
        nifti = self.load_nifti(load_path)
        nifti_pet = self.load_nifti(load_path2)
        numpy = self.nifti2numpy(nifti)
        norm = self.prep_numpy(numpy, mode = mode)
        
        if str(mode) == 'ct':
            print('ct')
            self.transform_nifti(nifti, nifti_pet, norm, save_path)
        else:
            self.save_nifti(nifti, norm, save_path)
        
    
    def load_data(self):
        print('\nLoading nifti files...')

        patients = self.summary

        #TODO: take filenames as arguments...
        for patient in tqdm(patients):     
            print(patient)
            # Ignore the pickle file in the data directory
            if not 'pickle' in str(patient):
                hd = self.create_paths(patient, self.hd_name)
                ld = self.create_paths(patient, self.ld_name)
                ct = self.create_paths(patient, self.ct_name)
            # ld[0] used for affine CT transformation
            #self.create_new_nifti(hd[0], ld[0], hd[1], 'hd')
            #self.create_new_nifti(ld[0], ld[0], ld[1], 'ld') 
            self.create_new_nifti(ct[0], ld[0], ct[1], 'ct')