#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 19 2020 by michellef

"""
import argparse
import rb82_model as rb82
import data_generator as data
import matplotlib as mpl
mpl.use('Agg')
TF_CPP_MIN_LOG_LEVEL = 2


if __name__ == "__main__":

    # Data paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', dest='data_path',
                        default='/homes/michellef/my_projects/rb82_data/Dicoms_OCT8', help="dicom file directory")
    parser.add_argument('--ld_path', dest='ld_path', default='25p_STAT', help='low dose PET folder name')
    parser.add_argument('--hd_path', dest='hd_path', default='100p_STAT', help='high dose PET folder name')

    # Image parameters
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=16, help='number of slices in a patch')
    parser.add_argument('--image_size', dest='image_size', type=int, default=128, help='image whole size')

    # Network training arguments
    parser.add_argument('--model_name', dest='model_name', default='Rb82_denoise', help='model name')
    parser.add_argument('--learning_rate', '-l', dest='lr', type=float, default=0.000001, help='learning rate')
    parser.add_argument('--epoch', '-e', dest='epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--initial_epoch', dest='initial_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--input_channels', dest='input_channels', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_channels', dest='output_channels', type=int, default=1, help='number of input channels')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--augment', '-a', dest='augment', type=data.ParseBoolean,
                        default=True, help='apply data augmentation: true, false')
    # Resume previous training
    parser.add_argument('--continue_train', dest='continue_train', type=data.ParseBoolean,
                        default=False, help='resume training: true, false')
    
    # Choose network version
    parser.add_argument('--version', dest='version', type=int, default=1, help = '1,2,3...')
    
    # PET normalisation number
    parser.add_argument('--norm', dest='norm', type=float, default=65535.0, help = 'PET normalisation factor')

    parser.add_argument('--phase', dest='phase', default='train', help='train or test')
    parser.add_argument('--kfold', dest='kfold', type=int, default='0', help='fold number')
    parser.add_argument('--maxp', dest='maxp', type=int, help='maximum number of patient to process')
    
    # Learning Rate Scheduler
    parser.add_argument('--lrs', dest='lrs', type=data.ParseBoolean,
                        default=True, help='use LRS: true, false')

    args = parser.parse_args()

    model = rb82.NetworkModel(args)
    model.predict() if args.phase == 'test' else model.train()
