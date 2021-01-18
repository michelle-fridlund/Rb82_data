#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:57:19 2018

@author: claesnl
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, add
from tensorflow.keras import backend as K, regularizers
import os
import PETCT_inference

def conv_block(layer,fsize,dropout,downsample=True):
    for i in range(1,3):
        layer = Conv3D(fsize,(3,3,3), padding='same',strides=(1,1,1))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(dropout)(layer)
    if downsample:
        downsample = Conv3D(fsize*2, (3,3,3), padding='same', strides=(2,2,2))(layer)
        downsample = BatchNormalization()(downsample)
        downsample = Activation('relu')(downsample)
    return layer, downsample

def convt_block(layer, concat, fsize):
    layer = Conv3DTranspose(fsize, (3,3,3), padding='same', strides=(2,2,2))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = concatenate([layer, concat], axis=-1)
    return layer

# Filters like Chen et al.
def construct_3D_unet_architecture(X):
    
    # ENCODING
    block1, dblock1 = conv_block(X,64,.1) # 64x64x16
    block2, dblock2 = conv_block(dblock1,128,.1) # 32x32x8
    block3, dblock3 = conv_block(dblock2,256,.2) # 16x16x4
    block4, _ = conv_block(dblock3,512,.3,downsample=False)

    # DECODING
    block5 = convt_block(block4,block3,256) 
    block6, _ = conv_block(block5,256,.2,downsample=False)

    block7 = convt_block(block6,block2,128)
    block8, _ = conv_block(block7,128,.2,downsample=False)

    block9 = convt_block(block8,block1,64)
    block10, _ = conv_block(block9,64,.1,downsample=False)

    output = Conv3D(1,(3,3,3), padding='same',strides=(1,1,1), activation='relu')(block10)
    
    return output

def prepare_3D_unet(x,y,z,d,initialize_model=None,classification=False,lr=0.0001,loss='mean_absolute_error'):

    inp = Input(shape=(x,y,z,d))
    output = construct_3D_unet_architecture(inp)
    model = Model(inputs=inp, outputs=output)

    if initialize_model:
        if not initialize_model.endswith('h5'):
            initialize_model = initialize_model+'.h5'
            
        #initialize_model = initialize_model if not initialize_model.endswith('.h5') else initialize_model+'.h5'
        if os.path.exists(initialize_model):
            # load weights
            model.load_weights(initialize_model)
            print("Created model and loaded weights from file: %s" % initialize_model)
        else:
            exit('did not find model %s' % initialize_model)

#    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[rmse,dice,'mean_absolute_error','acc'])
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['mean_absolute_error','acc'])
    return model

