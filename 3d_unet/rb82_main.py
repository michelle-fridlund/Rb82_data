#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:24:33 2018

Edited on Jan 19 2020 by michellef

@author: claesnl
"""

import os, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib as mpl
mpl.use('Agg')
import net_v1 as net # for residual - no relu at the end
import data_generator as data
import rb82_model as model 
import argparse
TF_CPP_MIN_LOG_LEVEL=2


if __name__=="__main__":
    
    #Data paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', dest='data_path', default='/homes/michellef/my_projects/Rb82/data/Dicoms_OCT8', help="dicom file directory")
    parser.add_argument('--ld_path', dest='ld_path', default='50p_STAT', help='low dose PET folder name')
    parser.add_argument('--hd_path', dest='hd_path', default='100p_STAT', help='high dose PET folder name')
    parser.add_argument('--state', dest='state_name', default='REST', type=data.Capitalise, help='REST or STRESS')
    
    #Image parameters
    parser.add_argument('--patch_size', dest='patch_size', type=int, default = 64, help='image patch size')
    parser.add_argument('--image_size', dest='image_size', type=int, default = 128, help='image whole size')
    
    #Network training arguments
    parser.add_argument('--model', dest='model', default ='Rb82_denoise', help='model name')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default = 1, help='batch size')
    parser.add_argument('--learning_rate', '-l', dest='lr', type=int, default = 0.0001, help='learning rate')
    parser.add_argument('--epoch', '-e', dest='epoch', type=int, default = 100, help='number of epochs')

    parser.add_argument('--augment', '-a', dest='augment', type=data.ParseBoolean, default = False, help='apply data augmentation: true, false')
    
    parser.add_argument('--train_or_test', dest='train_or_test', default='train', help='train or test')
    
    args = parser.parse_args()
    
    #model.NetworkModel.do_train(args) if args.train_or_test == 'train' else model.NetworkModel.do_test(args)
    data.DCMDataLoader(args)