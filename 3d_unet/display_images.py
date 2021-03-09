#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:48 2021

@author: michellef
"""
import os
import ants
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt

# img = ants.image_read("/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/Rb82_denoise_e100_bz1_lr0.0001_k0_predicted/0941d97c-f7b0-425b-b8b6-fa2e6f6ca595/0941d97c-f7b0-425b-b8b6-fa2e6f6ca595_predicted.nii.gz")

# tgt = ants.image_read("/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/25p_STAT/0941d97c-f7b0-425b-b8b6-fa2e6f6ca595/3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz")

# ants.plot(tgt)

img = nib.load("/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/100p_STAT/0941d97c-f7b0-425b-b8b6-fa2e6f6ca595/3_rest-lm-00-psftof_000_000_ctmv_4i_21s.nii.gz")
img2 = np.array(img.get_fdata(),dtype='double')

(a,b,c) = img2.shape
print(a)
#plt.imshow(img2[:,:,50])