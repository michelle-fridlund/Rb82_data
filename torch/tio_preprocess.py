import torchio as tio
from torchio import ScalarImage
from torchio.transforms import Resample
import nibabel as nib
import numpy as np
import os

def find_ct_patients():
    __data_path = '/homes/michellef/my_projects/rhtorch/my-torch/quadra/quadra_norm'
    _file_dict = {}
    # Train patient only
    patients = os.listdir('/homes/michellef/my_projects/rhtorch/quadra/static')
    for p in patients:
        patient_path = os.path.join(__data_path, p)
        # Reference PET path
        pet_path = os.path.join(patient_path, 'static_600s_suv_suv.nii.gz')
        # Original CT path
        ct_path = os.path.join(patient_path, 'ct_mask_affine.nii.gz')
        _file_dict[patient_path] = ((pet_path,ct_path))
    return _file_dict

def load_data(pet_path, ct_path):
    #reference PET image
    pet_data = ScalarImage(pet_path)
    ct_data = ScalarImage(ct_path)
    transform = tio.Resample(pet_data)
    fixed = transform(ct_data)
    return fixed 

def save_nifty(data, ref, filename_out):
    func = nib.load(ref).affine
    data = data.numpy().squeeze()
    print(np.shape(data))
    ni_img = nib.Nifti1Image(data, func)
    nib.save(ni_img, filename_out)

def preprocess_ct():
    ct_dict = find_ct_patients()

    for k,v in ct_dict.items():
        pet_path, ct_path = v
        fixed_ct = load_data(pet_path, ct_path)
        filename_out = os.path.join(k,'ct_mask_affine_resample.nii.gz')
        save_nifty(fixed_ct, ct_path, filename_out)
        print(f'{os.path.basename(k)} done.')

if __name__ == "__main__":
    preprocess_ct()