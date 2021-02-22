"""
Jan 19 12:59

michellef
##############################################################################

##############################################################################
"""

import net_v1 as net
import data_generator as data
import matplotlib as mpl
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import os
import glob
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

        self.kfold = args.kfold
        self.train_pts = 'train_{}'.format(self.kfold)
        self.valid_pts = 'valid_{}'.format(self.kfold)
        self.test_pts = 'test'

        # image params
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        # training params
        self.lr = args.lr
        self.epoch = args.epoch
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size

        self.n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary['train_0'])
        self.n_batches /= self.batch_size

        self.model_outname = str(args.model_name) + "_e" + \
            str(self.epoch) + "_bz" + str(self.batch_size) + "_lr" + \
            str(self.lr)+"_k" + str(self.kfold)

    def mkdir_(self, output):
        if not os.path.exists(output):
            os.makedirs(output)

    def load_data(self, mode):
        args = self.args
        loader = data.DCMDataLoader(args, mode)
        return loader.load_train_data(mode)

    def yield_data(self, stack):
        for value in stack.values():
            for state, pair in value.items():
                if state == 'UNKNOWN':
                    print('Patient state for selected pair is unknown, skipping...')
                    continue
                (ld, hd) = pair
                x = ld[-1, :, :, :]
                y = hd[-1, :, :, :]
                yield x, y

    def generator_train(self):
        stack = self.load_data(self.train_pts)
        return self.yield_data(stack)

    def generator_validate(self):
        stack = self.load_data(self.valid_pts)
        return self.yield_data(stack)

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

        self.mkdir_('checkpoint/TB/{}'.format(model_outname))
        filepath = os.path.join(
            'checkpoint', model_outname + "_e{epoch:03d}.h5")
        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_freq='epoch')  # use save_freq

        tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname),
                                 histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)
        callbacks_list = [checkpoint, tbCallBack]

        # Train model on dataset
        self.model.fit(self.data_train_gen,
                       steps_per_epoch=self.n_batches*(self.image_size**2//self.patch_size**2)//self.batch_size,
                       validation_data=self.data_valid_gen,
                       validation_steps=100,
                       epochs=self.epoch,
                       verbose=1,
                       callbacks=callbacks_list,
                       initial_epoch=initial_epoch)

        # Save model
        self.model.save(model_outname+".h5")
        print("Saved model to disk")
        
    def model_test(self):
        stack = self.load_data(self.test_pts)
        print(stack)
        return
        for value in stack.values():
            for state, pair in value.items():
                if state == 'UNKNOWN':
                    print('Patient state for selected pair is unknown, skipping...')
                    continue
                (ld, hd) = pair
                x = ld[-1, :, :, :]
                print(x)
                    
    # def train(self,args,LOG=None,MULTIGPU=False):
    def train(self):

        if os.path.exists(self.model_outname+".h5"):
            print("Model %s exists" % self.model_outname)

        # Initialise training
        self.model_train(self.model_outname, 128, 128, 16, 2,
                         self.data_path, self.epoch, self.batch_size, self.lr)
        
    # TODO: Implement resume training / transfer learning 

    # def test(self, args, x=args.image_size,y=dims_inplane,z=stack_of_slices,d=2,LOG=None):
    def test(self):
        if os.path.exists(f'{self.model_outname}.h5'):
            model_name = self.model_outname + '.h5'
        else:
            checkpoint_models = glob.glob('checkpoint/{}*.h5'.format(self.model_outname))
            if not checkpoint_models:
                print('No pretrained models found')
                exit(-1)
            model_name = checkpoint_models[-1]
        print(model_name)
        #model = load_model(model_name)

        # model_predict(model,model_name,x,y,z,d,LOG=LOG,orientation='axial')
