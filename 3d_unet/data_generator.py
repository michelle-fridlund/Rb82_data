"""
Jan 18 15:56

michellef
##############################################################################
Upload 3D PET data
##############################################################################
"""

import nibabel as nib
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class MissingNiftiData(Exception):
    """Raised when there is no nifti data for a patient"""
    pass


class DCMDataLoader(object):

    def __init__(self, args, summary, mode):

        # List of patients (loaded in model)
        self.summary = summary

        # paths to dicoms files
        self.data_path = args.data_path
        self.ld_path = args.ld_path
        self.hd_path = args.hd_path

        # image params
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.phase = args.phase

        # training params
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size

        # data augmentation
        self.augment = args.augment
        self.augmentation_params = {  # 'rotation range': [5, 5 ,5],
            'shift_range': [0.05, 0.05, 0.05],
            'shear_range': [2, 2, 0],
            'zoom_lower': [0.9, 0.9, 0.9],
            'zoom_upper': [1.2, 1.2, 1.2],
            'zoom_independent': True,
            'flip_axis': [1, 2],
            # 'X_shift_voxel' :[2, 2, 0],
            # 'X_add_noise' : 0.1,
            'fill_mode': 'reflect'
        }

    # Create a dict with filename as a key and numpy array as a value

    def load_nifti(self, path):
        return {os.path.basename(i): self.nifti2numpy(nib.load(i)) for i in glob.glob("{}/*.nii.gz".format(path), recursive=True)}

    # Transform DICOMS into numpy
    def nifti2numpy(self, nifti):
        try:
            return np.array(nifti.get_fdata(), dtype=np.float16)
        except:
            return None

    def augment_data(self, x, y):
        from DataAugmentation3D import DataAugmentation3D

        augment3D = DataAugmentation3D(**self.augmentation_params)
        x, y = augment3D.random_transform_batch(x, y)
        return x, y

    def load_train_data(self, mode):
        print('Loading nifti files...')

        patients = self.summary[mode]
        stack_dict = {}
        # Load and reshape all patient data
        for patient in patients:
            # Print progress
            print('.', end='', flush=True)

            stack_dict[patient] = {}
            ld_data = self.load_nifti('%s/%s/%s' % (self.data_path, self.ld_path, patient))
            hd_data = self.load_nifti('%s/%s/%s' % (self.data_path, self.hd_path, patient))

            try:
                if not ld_data or not hd_data:
                    raise MissingNiftiData()
            except MissingNiftiData:
                print(f'No nifti files for patient {patient} found')
                continue

            for key, value in ld_data.items():
                # Find low and high dose pairs by filenames
                lowres = value
                hires = hd_data.get(key, None)

                # rest or stress
                patient_state = 'UNKNOWN'
                if 'rest' in key.lower():
                    patient_state = 'REST'
                elif 'stress' in key.lower():
                    patient_state = 'STRESS'

                try:
                    if not isinstance(hires, np.ndarray) or not isinstance(lowres, np.ndarray):
                        raise MissingNiftiData()
                except MissingNiftiData:
                    print(f'A nifti pair for patient {patient} is missing')
                    continue

                # print()  # Blank line
                # print(patient_state)
                # print(key)
                # print(lowres.shape)
                # print(hires.shape)

                ld_ = lowres.reshape(128, 128, 111, 1)
                hd_ = hires.reshape(128, 128, 111, 1)
                
                # Determine slice
                z = np.random.randint(8, 111-8, 1)[0]
                ld_ = ld_[:, :, z-8:z+8, :]
                hd_ = hd_[:, :, z-8:z+8, :]

                x = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size)
                             + (self.input_channels,))
                y = np.empty((self.batch_size,) + (self.image_size, self.image_size, self.patch_size)
                             + (self.output_channels,))

                for i in range(self.batch_size):
                    x[i, ...] = ld_
                    y[i, ...] = hd_.reshape((self.image_size, self.image_size, self.patch_size)
                                            + (self.output_channels,))

                    if self.phase == 'train' and self.augment:
                        x, y = self.augment_data(x, y)

                if stack_dict.get(patient, {}).get(patient_state):
                    print(f'There are more nifti files for patient {patient} than needed. Skipping this patient...')
                    stack_dict.pop(patient, None)
                    break

                stack_dict[patient][patient_state] = (ld_, hd_)

        print()
        print('Finished loading nifti files')
        return stack_dict


# main.py parsers
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


def Capitalise(s):
    return s.upper()
