"""
Jan 19 12:59

michellef
##############################################################################

##############################################################################
"""

import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import glob
from keras.models import load_model, model_from_json
import os, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib as mpl
mpl.use('Agg')
import net_v1 as net # for residual - no relu at the end
import data_generator as data
import argparse
TF_CPP_MIN_LOG_LEVEL=2

dims_inplane = 128
stack_of_slices = 16

class NetworkModel(object):
    
    def mkdir_(self, output):
        if not os.path.exists(output):
            os.makedirs(output)

    def load_data(self, args, mode):
          
          ld,hd = data.DCMDataLoader.load_train_data(args, mode)
          
          x = ld.reshape((128,128,16,1))
          y = hd.reshape((128,128,16,1))
          return x, y
    
    def generator_train(self, args):
        yield self.load_data(args, 'train')
            
    def generator_validate(self, args):
        yield self.load_data(args, 'valid')
        
    def model_train(self,args,model_outname,x,y,z,d,data_path,epoch,batch_size, \
                    lr,verbose=1,train_pts=None,validate_pts=None,initial_epoch=0,\
                    initial_model=None,MULTIGPU=False,loss="mae"):
        # Generators
        data_train_gen = tf.data.Dataset.from_generator(self.generator_train, \
                                                        output_types=(tf.float32,tf.float32), \
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 1)),tf.TensorShape((128, 128, 16, 1))))
        data_valid_gen = tf.data.Dataset.from_generator(self.generator_validate, \
                                                        output_types=(tf.float32,tf.float32), \
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 1)),tf.TensorShape((128, 128, 16, 1))))
        
        data_valid_gen = data_valid_gen.repeat().batch(args.batch_size)
        data_train_gen = data_train_gen.repeat().batch(args.batch_size)
        
        model = net.prepare_3D_unet(x,y,z,d,initialize_model=initial_model,lr=lr,loss=loss)
        
        self.mkdir_('checkpoint')
        filepath = os.path.join('checkpoint', args.model_outname + "_e{epoch:03d}.h5")
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_freq='epoch') # use save_freq
        self.mkdir_('checkpoint/TB')
        self.mkdir_('checkpoint/TB/{}'.format(model_outname))
        
        tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname), histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)
        
        callbacks_list = [checkpoint,tbCallBack]
        
        num_pts = 77
        stack_of_slices = 16
        patches_pr_patient=128-stack_of_slices
        
        # Train model on dataset
        model.fit(data_train_gen,
                  steps_per_epoch = num_pts*patches_pr_patient//args.batch_size,
                  validation_data = data_valid_gen,
                  validation_steps = 100, 
                  epochs=args.epoch,
                  verbose=1,
                  callbacks=callbacks_list,
                  initial_epoch=initial_epoch)
    
        ## SAVE MODEL
        model.save(model_outname+".h5")
        print("Saved model to disk")
        
    def do_train(self,args,LOG=None,MULTIGPU=False): 

        self.model_outname = args.model+"_e"+str(args.epoch)+"_bz"+str(args.batch_size)+"_lr"+str(args.lr)

        if os.path.exists(self.model_outname+".h5"):
            print("Model %s exists" % self.model_outname)
            return
        
        # Initialize training
        self.model_train(self.model_outname,128,128,16,1,args.data_folder,args.epoch,args.batch_size,args.lr)

    def do_test(self, x=dims_inplane,y=dims_inplane,z=stack_of_slices,d=2,LOG=None):

        if os.path.exists(f'{self.model_outname}.h5'):
            model_name = self.model_outname+'.h5'
        else:
            model_name_cps = glob.glob(f'checkpoint/{model_name}*.h5')
            model_name = model_name_cps[-1]
    
        model = load_model(model_name)
    
        model_predict(model,model_name,x,y,z,d,LOG=LOG,orientation='axial')
        
