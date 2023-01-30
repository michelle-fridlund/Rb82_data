'''Mon Nov 21 
Script to sort Rb82 test set 
for MASKED VOI image metric calculations'''

import os
import dicom2nifti
from tqdm import tqdm
import numpy as np 
import argparse
import nibabel as nib
from shutil import copy
from pathlib import Path
# Below is for pytorch2 conda environment 
#from torchio import ScalarImage
#from torchio.data import LabelSampler
#from torchio.transforms import Resample


patients = ['44ae9693-b477-11ec-b751-e3702ec34f99','44ae96df-b477-11ec-b751-e3702ec34f99','44ae96af-b477-11ec-b751-e3702ec34f99','44ae9703-b477-11ec-b751-e3702ec34f99','44ae967d-b477-11ec-b751-e3702ec34f99','44ae96bc-b477-11ec-b751-e3702ec34f99','44ae96e0-b477-11ec-b751-e3702ec34f99','44ae96c6-b477-11ec-b751-e3702ec34f99','44ae9710-b477-11ec-b751-e3702ec34f99','44ae96a3-b477-11ec-b751-e3702ec34f99','44ae971d-b477-11ec-b751-e3702ec34f99','44ae9713-b477-11ec-b751-e3702ec34f99','44ae9711-b477-11ec-b751-e3702ec34f99','44ae96e7-b477-11ec-b751-e3702ec34f99','44ae96ee-b477-11ec-b751-e3702ec34f99','44ae96a4-b477-11ec-b751-e3702ec34f99','44ae9678-b477-11ec-b751-e3702ec34f99','44ae96f6-b477-11ec-b751-e3702ec34f99','44ae9689-b477-11ec-b751-e3702ec34f99','44ae968e-b477-11ec-b751-e3702ec34f99','44ae967e-b477-11ec-b751-e3702ec34f99','44ae9675-b477-11ec-b751-e3702ec34f99','44ae96c2-b477-11ec-b751-e3702ec34f99','44ae96cc-b477-11ec-b751-e3702ec34f99','44ae96f1-b477-11ec-b751-e3702ec34f99','44ae96a7-b477-11ec-b751-e3702ec34f99','44ae96b6-b477-11ec-b751-e3702ec34f99','44ae968a-b477-11ec-b751-e3702ec34f99','44ae9676-b477-11ec-b751-e3702ec34f99','44ae9680-b477-11ec-b751-e3702ec34f99','44ae9687-b477-11ec-b751-e3702ec34f99','44ae9694-b477-11ec-b751-e3702ec34f99','44ae9717-b477-11ec-b751-e3702ec34f99','44ae96f7-b477-11ec-b751-e3702ec34f99','44ae971e-b477-11ec-b751-e3702ec34f99','44ae96da-b477-11ec-b751-e3702ec34f99','44ae9720-b477-11ec-b751-e3702ec34f99','44ae9695-b477-11ec-b751-e3702ec34f99','44ae9714-b477-11ec-b751-e3702ec34f99','44ae967f-b477-11ec-b751-e3702ec34f99','44ae96f9-b477-11ec-b751-e3702ec34f99','44ae96f3-b477-11ec-b751-e3702ec34f99','44ae9679-b477-11ec-b751-e3702ec34f99','44ae9690-b477-11ec-b751-e3702ec34f99','44ae96e2-b477-11ec-b751-e3702ec34f99','44ae96e9-b477-11ec-b751-e3702ec34f99','44ae9704-b477-11ec-b751-e3702ec34f99','44ae96e1-b477-11ec-b751-e3702ec34f99','44ae969f-b477-11ec-b751-e3702ec34f99','44ae96e5-b477-11ec-b751-e3702ec34f99','44ae9729-b477-11ec-b751-e3702ec34f99','44ae9716-b477-11ec-b751-e3702ec34f99','44ae9672-b477-11ec-b751-e3702ec34f99','44ae96bf-b477-11ec-b751-e3702ec34f99','44ae96fc-b477-11ec-b751-e3702ec34f99','44ae969e-b477-11ec-b751-e3702ec34f99','44ae9696-b477-11ec-b751-e3702ec34f99','44ae9727-b477-11ec-b751-e3702ec34f99','44ae96b2-b477-11ec-b751-e3702ec34f99','44ae96d8-b477-11ec-b751-e3702ec34f99','44ae96c4-b477-11ec-b751-e3702ec34f99','44ae96a8-b477-11ec-b751-e3702ec34f99','44ae9681-b477-11ec-b751-e3702ec34f99','44ae96c0-b477-11ec-b751-e3702ec34f99','44ae9698-b477-11ec-b751-e3702ec34f99','44ae96d9-b477-11ec-b751-e3702ec34f99','44ae9700-b477-11ec-b751-e3702ec34f99','44ae96d6-b477-11ec-b751-e3702ec34f99','fba14f66-b428-11ec-b751-e3702ec34f99','44ae967a-b477-11ec-b751-e3702ec34f99','44ae96ed-b477-11ec-b751-e3702ec34f99','44ae9697-b477-11ec-b751-e3702ec34f99','44ae9706-b477-11ec-b751-e3702ec34f99','44ae970f-b477-11ec-b751-e3702ec34f99','44ae96d4-b477-11ec-b751-e3702ec34f99','44ae969c-b477-11ec-b751-e3702ec34f99','44ae9686-b477-11ec-b751-e3702ec34f99','44ae96e6-b477-11ec-b751-e3702ec34f99','44ae9677-b477-11ec-b751-e3702ec34f99','44ae9705-b477-11ec-b751-e3702ec34f99','44ae96a6-b477-11ec-b751-e3702ec34f99','44ae971f-b477-11ec-b751-e3702ec34f99','44ae96ad-b477-11ec-b751-e3702ec34f99','44ae96d7-b477-11ec-b751-e3702ec34f99','44ae972c-b477-11ec-b751-e3702ec34f99','44ae96a9-b477-11ec-b751-e3702ec34f99','44ae9671-b477-11ec-b751-e3702ec34f99','44ae9674-b477-11ec-b751-e3702ec34f99','44ae96c9-b477-11ec-b751-e3702ec34f99','44ae96a1-b477-11ec-b751-e3702ec34f99','44ae9718-b477-11ec-b751-e3702ec34f99','44ae96fb-b477-11ec-b751-e3702ec34f99','44ae9728-b477-11ec-b751-e3702ec34f99','44ae968b-b477-11ec-b751-e3702ec34f99','44ae968d-b477-11ec-b751-e3702ec34f99','44ae970d-b477-11ec-b751-e3702ec34f99','44ae972d-b477-11ec-b751-e3702ec34f99','44ae96e8-b477-11ec-b751-e3702ec34f99','44ae9692-b477-11ec-b751-e3702ec34f99','44ae9702-b477-11ec-b751-e3702ec34f99','44ae96a0-b477-11ec-b751-e3702ec34f99','44ae9685-b477-11ec-b751-e3702ec34f99','44ae972f-b477-11ec-b751-e3702ec34f99','44ae968c-b477-11ec-b751-e3702ec34f99','44ae972b-b477-11ec-b751-e3702ec34f99','44ae9708-b477-11ec-b751-e3702ec34f99']

mask_origin = '/homes/michellef/my_projects/ct_thorax/rb82_test_ct'
pet_origin = '/homes/michellef/Rb82_test_Sep2/output_fixed/static_out_dcm_FLIPPED'

# HD
pet_origin_r = '/homes/michellef/Rb82_test_Sep2/hd_static/rest/REST_100p_6p5mm'
pet_origin_s = '/homes/michellef/Rb82_test_Sep2/hd_static/stress/STRESS_100p_6p5mm'
output_dir = '/homes/michellef/Rb82_test_Sep2/NOV21_masked_test_metrics/static'

def dicom_to_nifti(input_, output_):
    if not os.path.exists(output_):
        os.makedirs(output_)
    dicom2nifti.convert_directory(input_, output_)

def transform_to_nifti(args):
    for p in tqdm(patients):
        input_r = f'{pet_origin}/{p}_rest'
        input_s = f'{pet_origin}/{p}_stress'
        output_r = f'{output_dir}/{p}_rest'
        output_s = f'{output_dir}/{p}_stress'

        if args.convert:
            dicom_to_nifti(input_r, output_r)
            dicom_to_nifti(input_s, output_s)

def transform_to_nifti_hd(args):
    for p in tqdm(patients):
        input_r = f'{pet_origin_r}/{p}'
        input_s = f'{pet_origin_s}/{p}'
        output_r = f'{output_dir}/{p}_rest'
        output_s = f'{output_dir}/{p}_stress'

        if args.convert:
            dicom_to_nifti(input_r, output_r)
            dicom_to_nifti(input_s, output_s)


def rename_nifti():
    for p in tqdm(patients):
        old_r = f'{output_dir}/{p}_rest/3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'
        old_s = f'{output_dir}/{p}_stress/3_stress-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz'
        new_r = f'{output_dir}/{p}_rest/pet_hd.nii.gz'
        new_s = f'{output_dir}/{p}_stress/pet_hd.nii.gz'

        os.rename(old_r, new_r)
        os.rename(old_s, new_s)


def nifti2numpy(nifti): 
        try: 
            d_type = nifti.header.get_data_dtype() #Extract data type from nifti header 
            return np.array(nifti.get_fdata(), dtype=np.dtype(d_type)) 
        except: 
            return None    


# Below is for pytorch2 conda venv only!
def resample_data(pet_path, ct_path):
    #reference PET image
    pet_data = ScalarImage(pet_path)
    ct_data = ScalarImage(ct_path)
    transform = Resample(pet_data)
    fixed = transform(ct_data)
    return fixed    

def save_nifty(data, ref, filename_out):
    func = nib.load(ref).affine
    # Can't call a tensor menthod .numpy() on a numpy array
    data = data if isinstance(data, np.ndarray) else data.numpy().squeeze()
    if np.shape(data) != (128,128,111):
        print(filename_out)
    ni_img = nib.Nifti1Image(data, func)
    nib.save(ni_img, filename_out)
                  
def resample_ct(args):
    for p in tqdm(patients):
        input_ct = f'{mask_origin}/{p}.nii.gz'
        output_r = f'{output_dir}/{p}_rest'
        output_s = f'{output_dir}/{p}_stress'

        pet_r = os.path.join(output_r, 'pet.nii.gz')
        pet_s = os.path.join(output_s, 'pet.nii.gz')

        fixed_ct_r = resample_data(pet_r, input_ct)
        fixed_ct_s = resample_data(pet_s, input_ct)

        filename_out_r = os.path.join(output_r, 'ct.nii.gz')
        filename_out_s = os.path.join(output_s, 'ct.nii.gz')

        if args.convert:
            save_nifty(fixed_ct_r, input_ct, filename_out_r)
            save_nifty(fixed_ct_s, input_ct, filename_out_s)
        else: 
            print(filename_out_r + '\n' + filename_out_s)

def load_mask(pet_file, ct_file):
    nifti_pet = nib.load(pet_file)
    nifti_ct = nib.load(ct_file)
    numpy_pet = nifti2numpy(nifti_pet)
    numpy_ct = nifti2numpy(nifti_ct)

    # We are only interested in the area with value 2.0:
    ct_mask = np.where(numpy_ct == 2.0, 1.0, 0.0) 
    pet_ct_mask = numpy_pet*ct_mask

    return pet_ct_mask

def create_mask(args):
    for p in tqdm(patients):
        pet_r = f'{output_dir}/{p}_rest/pet_hd.nii.gz'
        pet_s = f'{output_dir}/{p}_stress/pet_hd.nii.gz'
        ct_r = f'{output_dir}/{p}_rest/ct.nii.gz'
        ct_s = f'{output_dir}/{p}_stress/ct.nii.gz'

        mask_r = load_mask(pet_r, ct_r)
        mask_s = load_mask(pet_s, ct_s)

        mask_out_r = os.path.join(Path(pet_r).parent, 'pet_ct_mask_hd.nii.gz')
        mask_out_s = os.path.join(Path(pet_s).parent, 'pet_ct_mask_hd.nii.gz')


        if args.convert:
            save_nifty(mask_r, pet_r, mask_out_r)
            save_nifty(mask_s, pet_s, mask_out_s)
        else:
            print(mask_out_r)
            print(mask_out_s)




if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    parser.add_argument('--convert', action='store_true',
                        help="convert dicom to nifti")

    args = parser.parse_args()

    create_mask(args)