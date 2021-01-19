#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:24:33 2018

@author: claesnl
"""

import os, glob, numpy as np
#import net_v23 as net
import net_v15 as net
import PETCT_data_6fold as data
from tensorflow.keras.models import load_model
import nibabel as nib
import matplotlib.pyplot as plt

TF_CPP_MIN_LOG_LEVEL=2

_lowdose_name = "reduced_default_spatiallynormalized_128x128x128.nii.gz"
_ct_name = "ct_normalized_spatiallynormalized_128x128x128.nii.gz"

dims_inplane = 128
stack_of_slices = 16
    
def do_predict_model_v1(x=dims_inplane,y=dims_inplane,z=stack_of_slices,d=2,LOG=None):
    
    model_name = 'PETCTPiB_v1_suv_100_bz4_lr0.0001'
    if os.path.exists(f'{model_name}.h5'):
        model_name = model_name+'.h5'
    else:
        model_name_cps = glob.glob(f'checkpoint/{model_name}*.h5')
        model_name = model_name_cps[-1]
    
    model = load_model(model_name,custom_objects={'rmse':net.rmse,'dice':net.dice}) #,'masked_RMSE':net.masked_RMSE
    
    model_predict(model,model_name,x,y,z,d,LOG=LOG,orientation='axial')

    
def model_predict_patient(pt,model,model_name,x,y,z,d,orientation='axial'):
    
    dat,res,pt_name = data.load_all_suv_ctnorm_double('valid', ind=pt)
    img = nib.load(f'data_michelle/{pt_name}/{_lowdose_name}')    
    
    model_name_ = model_name.split('/')[-1] if not model_name.endswith('.h5') else model_name.split('/')[-1][:-3]
    
    if os.path.exists(os.path.join('data_michelle',pt_name,'predicted_'+model_name_+'_'+_lowdose_name)):
        print(pt_name,"Skipping - output exists")
        return
    
    print("Predicting volume for %s" % pt_name)
    predicted = np.empty((128,128,128))
    
    for z_index in range(int(z/2),128-int(z/2)):       
        if orientation == 'coronal':
            
            
            # CORONAL
            predicted_stack = model.predict(dat[:,:,z_index-int(z/2):z_index+int(z/2),:].reshape(1,x,y,z,d))
            if z_index == int(z/2):
                for ind in range(int(z/2)):
                    predicted[:,:,ind] = predicted_stack[0,:,:,ind].reshape(128,128)
            if z_index == 128-int(z/2)-1:
                for ind in range(int(z/2)):
                    predicted[:,:,z_index+ind] = predicted_stack[0,:,:,int(z/2)+ind].reshape(128,128)
            predicted[:,:,z_index] = predicted_stack[0,:,:,int(z/2)].reshape(128,128) 
        
        elif orientation == 'axial':
            # AXIAL
            dat_stack = dat[:,z_index-int(z/2):z_index+int(z/2),:,:]
            dat_stack = np.swapaxes(dat_stack,1,2)
            predicted_stack = model.predict(dat_stack.reshape(1,x,y,z,d))
            #print(predicted_stack.shape)
            predicted_stack = np.swapaxes(predicted_stack,3,2)
            #print(predicted_stack.shape)
            assert predicted_stack.shape == (1,128,16,128,1)
            if z_index == int(z/2):
                for ind in range(int(z/2)):
                    predicted[:,ind,:] = predicted_stack[0,:,ind,:].reshape(128,128)
            if z_index == 128-int(z/2)-1:
                for ind in range(int(z/2)):
                    predicted[:,z_index+ind,:] = predicted_stack[0,:,int(z/2)+ind,:].reshape(128,128)
            predicted[:,z_index,:] = predicted_stack[0,:,int(z/2),:].reshape(128,128)
    
    predicted_full = predicted 
    predicted_image = nib.Nifti1Image(predicted_full, img.affine, img.header)
    nib.save(predicted_image,f'data_michelle/{pt_name}/predicted_{model_name_}_{_lowdose_name}')
    
def model_predict(model,model_name,x,y,z,d,ct_norm=False,LOG=None,orientation='axial'):
    
    # Predict patient
    for pt in data.get_summary('valid'):
        model_predict_patient(pt,model,model_name,x,y,z,d,orientation=orientation)
        #return
    
if __name__=="__main__":
    
    do_predict_model_v1()
