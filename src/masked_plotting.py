from rhscripts.plotting import plot_img_and_mask
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def nii2np(nifti):
    nifti_ = nib.load(nifti)
    d_type = nifti_.header.get_data_dtype()  # get data type from nifti header
    numpy_ = np.array(nifti_.get_fdata(), dtype=np.dtype(d_type))
    return numpy_

if __name__ == "__main__":
    image = nii2np('/homes/michellef/Rb82_test_Sep2/NOV21_masked_test_metrics/static/44ae9671-b477-11ec-b751-e3702ec34f99_rest/pet_hd.nii.gz')
    mask = nii2np('/homes/claes/data_shared/Rb82_test/ACCT_Segmentations/44ae9671-b477-11ec-b751-e3702ec34f99/heart_ventricle_left.nii.gz')
    mask = mask[::4,::4,]
    image = np.rot90(image)
    im = plt.imshow(image[:,:,55])
    #ax = plot_img_and_mask(image[:,:,55], mask[:,:,55])
    plt.savefig(f'/homes/michellef/image.png')