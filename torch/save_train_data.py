import os
import numpy as np
import argparse
import pickle
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from prep_torch_data import dicom_to_nifti
import pickle

req_files = os.listdir('/homes/claes/data_shared/Rb82_train/8cd25543-c065-11ec-b751-e3702ec34f99')

#req_files = ['random_gate_25p.nii.gz', 'random_gate_100p.nii.gz', 'static_100p.nii.gz', 'static_25p.nii.gz']

gated_files_ld = ['gate0_STRESS_25p.nii.gz', 'gate1_STRESS_25p.nii.gz', 'gate2_STRESS_25p.nii.gz', 'gate3_STRESS_25p.nii.gz', 'gate4_STRESS_25p.nii.gz', 'gate5_STRESS_25p.nii.gz', 'gate6_STRESS_25p.nii.gz', 'gate7_STRESS_25p.nii.gz', 'gate0_REST_25p.nii.gz', 'gate1_REST_25p.nii.gz', 'gate2_REST_25p.nii.gz', 'gate3_REST_25p.nii.gz', 'gate4_REST_25p.nii.gz', 'gate5_REST_25p.nii.gz', 'gate6_REST_25p.nii.gz', 'gate7_REST_25p.nii.gz']

gated_files_hd = ['gate0_STRESS_100p.nii.gz', 'gate1_STRESS_100p.nii.gz', 'gate2_STRESS_100p.nii.gz', 'gate3_STRESS_100p.nii.gz', 'gate4_STRESS_100p.nii.gz', 'gate5_STRESS_100p.nii.gz', 'gate6_STRESS_100p.nii.gz', 'gate7_STRESS_100p.nii.gz', 'gate0_REST_100p.nii.gz', 'gate1_REST_100p.nii.gz', 'gate2_REST_100p.nii.gz', 'gate3_REST_100p.nii.gz', 'gate4_REST_100p.nii.gz', 'gate5_REST_100p.nii.gz', 'gate6_REST_100p.nii.gz', 'gate7_REST_100p.nii.gz']


# Find all .nii.gz files in subdirs
def find_nifti(pt):
    pt = Path(pt)
    nifti = []
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if '.nii' in f.name:
            nifti.append(str(f))
    return nifti

# Find training patients with required files from predifined dir
def find_patients(args):
    data = args.data
    paths = {}
    patients = os.listdir(data)
    for p in patients:
    # Avoid any previously generated .pickle
        if '.pickle' not in p: 
            path = os.path.join(data, p)
            if data == '/homes/claes/data_shared/Rb82_train/':
                # Check if patient dir contains all required the files
                result = all(files in os.listdir(path) for files in req_files)
                if result:
                    # Return name and full path
                    paths[p] = path
            else:
                paths[p] = path
    return paths

def dcm2nifti(args):
    paths = find_patients(args)
    state = str(os.path.basename(args.data)).split('_')[0].lower()
    
    c = 0
    for k,v in paths.items():
        output_dir = f'/homes/michellef/my_projects/rhtorch/quadra/{k}_{state}'
        # Remove in case of redoing
        if args.force:
            try:
                os.remove(output_dir)
                print(f'deleted {k}')
            except Exception as error:
                print(error)
                print(f'Cannot delete {output_dir} or does not exist.')
                continue
        # check if the number of files is correct: static = 111, gated = 888        
        files = os.listdir(v)
        if len(files) == 0 or len(files) != 888:
            print(f'!No files found for {k}!')
        else:
            dicom_to_nifti(v, output_dir)
            #print(f'{c}. {input_} to {output_}')
        c += 1

    print(f'{c} patients converted.')

# Identify Quadra Recon Type
def find_recon_type(file_dir):
    # Subdirectory path for patient
    dcm_file = os.path.join(file_dir, os.listdir(file_dir)[0])
    os.chdir(file_dir)
    file_dir2 = Path(file_dir)
    if not (file_dir2/'dump.txt').exists():
        os.system(f"dcmdump '{dcm_file}' --search SeriesDescription >> \
                  '/homes/michellef/dump.txt'")

    with open('/homes/michellef/dump.txt') as f:
        for line in f.readlines():
            recon_type = line.strip()
    os.remove('/homes/michellef/dump.txt')
    return str(recon_type)

def write_pickle(data, save_path):
    save_path = Path(save_path)
    with open(save_path, 'wb') as p:
        pickle.dump(data, p)

def load_pickle(save_path):
    p = pickle.load(open(save_path,'rb'))
    return p

def sort_recon_type(args):
    paths = find_patients(args)

    if args.pickle:
        patient_recons = load_pickle('/homes/michellef/quadra_recons_Aug25.pickle')
    else:
        patient_recons = {}
        for k,v in paths.items():
            recon_types = {'90s': [],
                        '600s': [],
                        'CT': [],}

            subdirs = os.listdir(v)
            for s in subdirs:
                # Full path to subdirs
                full_r = os.path.join(v, s)
                recon_type = find_recon_type(full_r)
                if '90s' in str(recon_type):
                    recon_types['90s'].append(full_r)
                elif '600s' in str(recon_type):
                    recon_types['600s'].append(full_r)
                elif 'CT' in str(recon_type):
                    recon_types['CT'].append(full_r)
                else:
                    print(f'Unknown type at {k}')
        
                patient_recons[k] = recon_types
        write_pickle(patient_recons, '/homes/michellef/quadra_recons_Aug25.pickle')

    return patient_recons

def dcm2nifti_quadra(args):
    patient_recons = sort_recon_type(args)

    c = 0
    for k,v in patient_recons.items():
        if len(v['600s']) == 0:
            print(f'No files in {k}!!!')
        else:
            input_dir = v['600s'][0]
            output_dir = f'/homes/michellef/my_projects/rhtorch/quadra_600s/{k}'
            dicom_to_nifti(input_dir, output_dir)
            c+=1
            print(c)

#Unknown type at kj6XTeGGxR_0.0
#Unknown type at rZ2iIdEqx3_0.0
#Unknown type at Y8xxt2K8Zm_0.0
#Unknown type at BKUaYEujyW_0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help="Force file deletion before copying")
    parser.set_defaults(force=False)
    parser.add_argument('--data', dest='data', type=str, 
                        default='/homes/claes/data_shared/Rb82_train/',
                        help='data source directory')
    parser.add_argument("-q", "--quadra", help="data from quadra",
                        action="store_true")
    # Load patient dict from pickle
    parser.add_argument("-p", "--pickle", help="data from quadra",
                    action="store_true")
    # Parse this tag if original data in dicom format
    parser.add_argument("--dicom", help="plot values",
                        action="store_true")

    args = parser.parse_args()

    if args.quadra:
        dcm2nifti_quadra(args)
    else:
        #get_stats(args)
        dcm2nifti(args)
        #write_pickle_test()