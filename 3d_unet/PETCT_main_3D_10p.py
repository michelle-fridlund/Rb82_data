#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:24:33 2018

@author: claesnl
"""

import os, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib as mpl
mpl.use('Agg')
import net_v15 as net # for residual - no relu at the end
import data_generator as data
TF_CPP_MIN_LOG_LEVEL=2

dims_inplane = 128
stack_of_slices = 16
train_or_test = "train"
batch_size = -1
  
# Added correct patient weight, and now yielding axial slices! (v3)
def load_data_suv_ptweight(train_or_test,batch_size,orientation='axial'):
    
      augment = True if train_or_test == "train" else False
      
      dat,res = data.DCMDataLoader(data_path)
      
      X = dat.reshape((128,128,16,1))
      y = res.reshape((128,128,16,1))
    
      return X, y

def generator_train():
    yield load_data_suv_ptweight('train',batch_size,orientation='axial')
        
def generator_validate():
    yield load_data_suv_ptweight('valid',batch_size,orientation='axial')
    
def model_train(model_outname,x,y,z,d,data_folder='data_michelle',epochs=20,batch_size=1,lr=0.0001,verbose=1,train_pts=None,validate_pts=None,initial_epoch=0,initial_model=None,MULTIGPU=False,loss="mae"):

    # Generators
    data_train_gen = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32,tf.float32), output_shapes=(tf.TensorShape((128, 128, 16, 1)),tf.TensorShape((128, 128, 16, 1))))
    data_valid_gen = tf.data.Dataset.from_generator(generator_validate, output_types=(tf.float32,tf.float32), output_shapes=(tf.TensorShape((128, 128, 16, 1)),tf.TensorShape((128, 128, 16, 1))))
    
    # Use shuffle on dataset - not sure how/why it works and doesnt run out of memory
    data_valid_gen = data_valid_gen.repeat().batch(batch_size)
    data_train_gen = data_train_gen.repeat().batch(batch_size)
    
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    
    model = net.prepare_3D_unet(x,y,z,d,initialize_model=initial_model,lr=lr,loss=loss)
    
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    filepath=os.path.join('checkpoint',model_outname+"_e{epoch:03d}.h5")
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_freq='epoch') # use save_freq
    
    if not os.path.exists('checkpoint/TB'):
        os.makedirs('checkpoint/TB')
    if not os.path.exists('checkpoint/TB/{}'.format(model_outname)):
        os.makedirs('checkpoint/TB/{}'.format(model_outname))
    tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname), histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)
    
    callbacks_list = [checkpoint,tbCallBack]
    
    num_pts = 77
    patches_pr_patient=128-stack_of_slices
    # Train model on dataset
    model.fit(  data_train_gen,
                steps_per_epoch = num_pts*patches_pr_patient//batch_size,
                validation_data = data_valid_gen,
                validation_steps = 100, # 5 pts with SUV
                epochs=epochs,
                verbose=1,
                callbacks=callbacks_list,
                initial_epoch=initial_epoch)

    ## SAVE MODEL
    model.save(model_outname+".h5")
    print("Saved model to disk")
    
def do_train(LOG=None,MULTIGPU=False): # 
    # Model info
    model="PETCTPiB_v1_suv"
    epoch=100
    global batch_size
    batch_size=4 # 14 with 2 GPU crashes.
    global_batch_size = batch_size*2 if MULTIGPU else batch_size
    lr=0.0001
    model_outname = model+"_e"+str(epoch)+"_bz"+str(global_batch_size)+"_lr"+str(lr)
    
    if os.path.exists(model_outname+".h5"):
        print("Model %s exists" % model_outname)
        return
    
    # Initialize training
    model_train(model_outname,128,128,16,1,epochs=epoch,batch_size=batch_size,lr=lr)
    
if __name__=="__main__":
    do_train()