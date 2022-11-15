import os
import pickle
import math
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from pydicom import dcmread
from rhscripts.dcm import get_suv_constants

# Return a dict where name is key and gull path to CT is value
def read_pickle():
    ct_info = {}
    # Normalising only train patients
    patients = os.listdir('/homes/michellef/my_projects/rhtorch/quadra/static')
    patient_dict = pickle.load(open('/homes/michellef/quadra_recons_Aug25.pickle','rb'))

    for k, v in patient_dict.items():
        if k in patients:
            ct_dir = v['600s'][0]
            ct_file = os.listdir(ct_dir)[0]
            ct_info[k] = os.path.join(ct_dir, ct_file)
    return ct_info

def get_dicom(dcm_path):
    ds = dcmread(dcm_path)
    injection_time = ds.RadiopharmaceuticalStartTime
    return injection_time

def read_header():
    ct_info = read_pickle()

    patient = {
    'PatientWeight': [],
    'RadionuclideHalfLife': [],
    'RadionuclideTotalDose': [],
    'RadiopharmaceuticalStartTime': [],
    'PerformedProcedureStepStartTime': [],
    }
    for k,v in ct_info.items():
       #time = get_dicom(v)
       print(v)

def get_suv_info():
    ct_info = read_pickle()
    suv_transform_dict = {} 
    for k, v in ct_info.items():
        d, fn = get_suv_constants(v)
        suv_transform_dict[k] = (d, fn)
    return suv_transform_dict

def get_file_path(data_path, k):
    full = os.path.join(data_path,k)
    files = os.listdir(full)
    path_tuples = []
    for f in files:
        file_path = os.path.join(full,f)
        if '600s' in str(file_path):
            new_path = f'{full}/static_600s_suv.nii.gz'
        elif '90s' in str(file_path):
            new_path = f'{full}/static_90s_suv.nii.gz'
        else:
            print('Unknown recon type at {f}')
        path_tuples.append((file_path,new_path))
    return path_tuples

def nifti2numpy(nifti):
    try:
        d_type = nifti.header.get_data_dtype() #Extract data type from nifti header
        return np.array(nifti.get_fdata(), dtype=np.dtype(d_type))
    except:
        return None

def calclate_suv(d, numpy):
    return (numpy * d['weight']) / (d['corrected_dose'] * 1000)

def get_max(vals):
    return np.max(vals)

def norm_patients(data_path, args):
    suv_transform_dict = get_suv_info()
    max_values = []
    for k,v in suv_transform_dict.items():
        path_tuples = get_file_path(data_path,k)
        for tuples in path_tuples:
            old, new = tuples
            if args.mode == 'norm':
                nifti = nib.load(old)
                numpy = nifti2numpy(nifti)
                d, fn = v
                suv_numpy = calclate_suv(d, numpy)
                #vfunc = np.vectorize(fn)
                #norm = vfunc(numpy)
                save_nifti(nifti, suv_numpy, new)
            else:
                if '600s' in str(new):
                    nifti = nib.load(new)
                    numpy = nifti2numpy(nifti)
                    max = get_max(numpy)
                    max_values.append(max)
    mean_max = np.mean(max_values)
    print(f'Mean of max SUV values: {mean_max}')

    # Mean of max SUV values: 44.87

# Hardcoded for quadra ct files
# Return full path to original ct + save_path
def find_ct_patients():
    __data_path = '/homes/michellef/my_projects/rhtorch/quadra/ct'
    __new_path = '/homes/michellef/my_projects/rhtorch/my-torch/quadra/quadra_norm'
    file_tuples = []
    patients = os.listdir('/homes/michellef/my_projects/rhtorch/quadra/static')
    for p in patients:
        full_path = os.path.join(__data_path, p)
        ct_file = os.listdir(full_path)
        file_path = os.path.join(full_path, ct_file[0])
        new_path = os.path.join(__new_path, p, 'ct_mask_affine.nii.gz')
        file_tuples.append((file_path,new_path))
    return file_tuples

# Transform to HU (HU = x*slope - intercept)
def to_hu(ct,slope=1.0,intercept=-1024): 
     return ct*slope+intercept 

def save_nifti(nifti, numpy, save_path):
    image = nib.Nifti1Image(numpy, nifti.affine, nifti.header)
    nib.save(image, save_path)

def transform_nifti(nifti, nifti_pet, numpy, save_path):
    new_header = nifti.header.copy()
    # Use affine matrix to match PET input
    xform = nifti_pet.affine
    img = nib.Nifti1Image(numpy, xform, header = new_header)
    nib.save(img, save_path)

# Create binary masks, where
def norm_ct():
    ct_tuples = find_ct_patients()
    for tuples in ct_tuples:
            old, new = tuples
            __new_ = Path(new).parent.absolute()
            nifti_pet_path = os.path.join(__new_, 'static_600s_suv_suv.nii.gz')
            nifti_pet = nib.load(nifti_pet_path)

            nifti = nib.load(old)
            numpy = nifti2numpy(nifti)
            numpy_hu = to_hu(numpy)
            np_mask = np.where(numpy_hu > -500.0, 1.0, 0.0)

            #save_nifti(nifti, np_mask, new)
            transform_nifti(nifti, nifti_pet, np_mask, new)
            print(os.path.basename(new))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    required_args.add_argument(
        "--mode", dest='mode', help="mean/norm")

    data_path = '/homes/michellef/my_projects/rhtorch/my-torch/quadra/quadra_norm'
    args = parser.parse_args()
    #norm_patients(data_path, args)

    norm_ct()

    