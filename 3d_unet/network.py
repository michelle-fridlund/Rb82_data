#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:57:19 2018

@author: claesnl
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K, regularizers
import os


def conv_block(layer, fsize, dropout, downsample=True):
    for i in range(1, 3):
        layer = Conv3D(fsize, (3, 3, 3), kernel_regularizer=regularizers.l2(1e-1),
                       kernel_initializer='he_normal', padding='same', strides=(1, 1, 1))(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(dropout)(layer)
    if downsample:
        downsample = Conv3D(fsize*2, (3, 3, 3), kernel_regularizer=regularizers.l2(1e-1),
                            kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(layer)
        downsample = BatchNormalization()(downsample)
        downsample = Activation('relu')(downsample)
    return layer, downsample


def convt_block(layer, concat, fsize):
    layer = Conv3DTranspose(fsize, (3, 3, 3), kernel_regularizer=regularizers.l2(1e-1),
                            kernel_initializer='he_normal', padding='same', strides=(2, 2, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = concatenate([layer, concat], axis=-1)
    return layer

# Filters like Chen et al.


def construct_3D_unet_architecture(X):

    # ENCODING
    # block1, dblock1 = conv_block(X, 64, .1)  # 64x64x16
    # block2, dblock2 = conv_block(dblock1, 128, .1)  # 32x32x8
    # block3, dblock3 = conv_block(dblock2, 256, .2)  # 16x16x4
    # block4, _ = conv_block(dblock3, 512, .3, downsample=False)

    # # DECODING
    # block5 = convt_block(block4, block3, 256)
    # block6, _ = conv_block(block5, 256, .2, downsample=False)

    # block7 = convt_block(block6, block2, 128)
    # block8, _ = conv_block(block7, 128, .2, downsample=False)

    # block9 = convt_block(block8, block1, 64)
    # block10, _ = conv_block(block9, 64, .1, downsample=False)
    
    # ENCODING
    block1, dblock1 = conv_block(X, 32, .1)  #32 kernels  used
    block2, dblock2 = conv_block(dblock1, 64, .1)  
    block3, dblock3 = conv_block(dblock2, 128, .2)  
    block4, dblock4 = conv_block(dblock3, 256, .2)  
    block5, _ = conv_block(dblock4, 512, .3, downsample=False)

    block6 = convt_block(block5, block4, 256)
    block7, _ = conv_block(block6, 256, .3, downsample=False)

    block8 = convt_block(block7, block3, 128)
    block9, _ = conv_block(block8, 128, .2, downsample=False)

    block10 = convt_block(block9, block2, 64)
    block11, _ = conv_block(block10, 64, .2, downsample=False)
    
    block12 = convt_block(block11, block1, 32)
    block13, _ = conv_block(block12, 32, .1, downsample=False)
   
    output = Conv3D(1, (3, 3, 3), kernel_regularizer=regularizers.l2(1e-1),
                    kernel_initializer='he_normal', padding='same', 
                    strides=(1, 1, 1), activation='relu')(block13)

    return output

# def get_lr_metric(model):
#     def lr(y_true, y_pred):
#         optimizer = model.optimizer
#         iterations = optimizer.iterations
#         decay = optimizer.lr
#         lr = decay.decayed_learning_rate(iterations)
#         # decay = optimizer.lr
#         # return K.eval(lr.decay)
#         # # return K.eval(K.cast(decay, K.dtype(lr)))
#     return lr


def prepare_3D_unet(x, y, z, d, initialize_model=None, classification=False, lr=0.0001, loss='mean_absolute_error'):
    inp = Input(shape=(x, y, z, d))
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
            
    # initial_learning_rate = lr
    # # initial_learning_rate = tf.cast(initial_learning_rate, tf.float64)
    # lr_schedule = ExponentialDecay(
    # initial_learning_rate,
    # decay_steps=1000,
    # decay_rate=0.95)
    
    # lr_metric = get_lr_metric(model)
    # lr_metric = tf.cast(lr_metric, tf.float64)
    # model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mean_absolute_error', metrics=['mean_absolute_error', 'acc'])
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error', metrics=['mean_absolute_error', 'acc'])
    # print('Learning rate: ' + str(round(model.optimizer.lr.numpy(), 5)))
    return model
