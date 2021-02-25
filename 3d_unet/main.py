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
    #parser.add_argument('--state', dest='state_name', default='REST', type=data.Capitalise, help='REST or STRESS')

    # Image parameters
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=16, help='number of slices in a patch')
    parser.add_argument('--image_size', dest='image_size', type=int, default=128, help='image whole size')

    # Network training arguments
    parser.add_argument('--model_name', dest='model_name', default='Rb82_denoise', help='model name')
    parser.add_argument('--learning_rate', '-l', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', '-e', dest='epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--input_channels', dest='input_channels', type=int, default=2, help='number of input channels')
    parser.add_argument('--output_channels', dest='output_channels', type=int, default=1, help='number of input channels')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--augment', '-a', dest='augment', type=data.ParseBoolean,
                        default=True, help='apply data augmentation: true, false')

    parser.add_argument('--phase', dest='phase', default='train', help='train or test')
    parser.add_argument('--kfold', dest='kfold', type=int, default='0', help='fold number')
    parser.add_argument('--maxp', dest='maxp', type=int, help='maximum number of patient to process')

    args = parser.parse_args()

    model = rb82.NetworkModel(args)
    model.train() if args.phase == 'train' else model.model_predict()
