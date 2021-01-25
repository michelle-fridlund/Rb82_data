"""
Jan 19 12:59

michellef
##############################################################################

##############################################################################
"""

import net_v1 as net
import numpy as np
import data_generator as data
import matplotlib as mpl
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.models import load_model, model_from_json
import glob
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
mpl.use('Agg')
TF_CPP_MIN_LOG_LEVEL = 2


class NetworkModel(object):
    def __init__(self, args):
        # TODO: Rewrite?
        self.args = args
        # paths to dicoms files
        self.data_path = args.data_path
        # pickle file
        self.summary = pickle.load(
            open('%s/data.pickle' % self.data_path, 'rb'))

        # image params
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        # training params
        self.model_name = args.model_name
        self.lr = args.lr
        self.epoch = args.epoch
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size

        self.n_batches = len(self.summary['train'])
        self.n_batches /= self.batch_size

        #self.callbacks_list = self.setup_callbacks()

    def mkdir_(self, output):
        if not os.path.exists(output):
            os.makedirs(output)

    # def setup_callbacks(self):
    #     # Checkpoints
    #     os.makedirs('checkpoint/{}'.format(self.config['model_name']), exist_ok=True)
    #     checkpoint_file=os.path.join('checkpoint',self.config["model_name"],'e{epoch:02d}_{val_loss:.2f}.h5')
    #     checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False, mode='min',period=self.config["checkpoint_save_rate"])

    #     # Tensorboard
    #     os.makedirs('logs', exist_ok=True)
    #     TB_file=os.path.join('logs',self.config["model_name"])
    #     TB = TensorBoard(log_dir = TB_file)

    #     return [checkpoint, TB]

    def load_data(self, mode):
        args = self.args
        loader = data.DCMDataLoader(args, mode)
        ld, hd = loader.load_train_data(args, mode)

        X = ld[-1, :, :, :]
        y = hd[-1, :, :, :]
        # print(X.shape,y.shape)
        return X, y

    def generator_train(self):
        yield self.load_data('train')

    def generator_validate(self):
        yield self.load_data('valid')

    def model_train(self, model_outname, x, y, z, d, data_path, epoch, batch_size,
                    lr, verbose=1, train_pts=None, validate_pts=None, initial_epoch=0,
                    initial_model=None, MULTIGPU=False, loss="mae"):

        # Generators
        self.data_train_gen = tf.data.Dataset.from_generator(self.generator_train,
                                                             output_types=(
                                                                 tf.float32, tf.float32),
                                                             output_shapes=(tf.TensorShape((128, 128, 16, 2)), tf.TensorShape((128, 128, 16, 1))))
        self.data_valid_gen = tf.data.Dataset.from_generator(self.generator_validate,
                                                             output_types=(
                                                                 tf.float32, tf.float32),
                                                             output_shapes=(tf.TensorShape((128, 128, 16, 2)), tf.TensorShape((128, 128, 16, 1))))

        self.data_valid_gen = self.data_valid_gen.repeat().batch(self.batch_size)
        self.data_train_gen = self.data_train_gen.repeat().batch(self.batch_size)

        self.model = net.prepare_3D_unet(
            x, y, z, d, initialize_model=initial_model, lr=lr, loss=loss)

        self.mkdir_('checkpoint')
        filepath = os.path.join(
            'checkpoint', model_outname + "_e{epoch:03d}.h5")
        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_freq='epoch')  # use save_freq
        self.mkdir_('checkpoint/TB')
        self.mkdir_('checkpoint/TB/{}'.format(model_outname))

        tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname),
                                 histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)

        callbacks_list = [checkpoint, tbCallBack]

        # Train model on dataset
        self.model.fit(self.data_train_gen,
                       steps_per_epoch=self.n_batches *
                       (self.image_size-self.patch_size),
                       validation_data=self.data_valid_gen,
                       validation_steps=100,
                       epochs=self.epoch,
                       verbose=1,
                       callbacks=callbacks_list,
                       initial_epoch=initial_epoch)

        # Save model
        self.model.save(model_outname+".h5")
        print("Saved model to disk")

    # def train(self,args,LOG=None,MULTIGPU=False):
    def train(self):

        self.model_outname = self.model_name+"_e" + \
            str(self.epoch)+"_bz"+str(self.batch_size)+"_lr"+str(self.lr)

        if os.path.exists(self.model_outname+".h5"):
            print("Model %s exists" % self.model_outname)

        # Initialise training
        self.model_train(self.model_outname, 128, 128, 16, 2,
                         self.data_path, self.epoch, self.batch_size, self.lr)

    # def do_test(self, args, x=args.image_size,y=dims_inplane,z=stack_of_slices,d=2,LOG=None):

    #     if os.path.exists(f'{self.model_outname}.h5'):
    #         model_name = self.model_outname+'.h5'
    #     else:
    #         model_name_cps = glob.glob(f'checkpoint/{model_name}*.h5')
    #         model_name = model_name_cps[-1]

    #     model = load_model(model_name)

    #     #model_predict(model,model_name,x,y,z,d,LOG=LOG,orientation='axial')
