"""
Jan 19 12:59

michellef
##############################################################################

##############################################################################
"""

import data_generator as data
import matplotlib as mpl
import nibabel as nib
import numpy as np
import math
import pickle
import os
import glob
import warnings
warnings.filterwarnings('ignore')
mpl.use('Agg')
TF_CPP_MIN_LOG_LEVEL = 2


class NetworkModel(object):
    def __init__(self, args):

        self.args = args
        # paths to dicoms files
        self.data_path = args.data_path
        self.ld_path = args.ld_path
        # pickle file
        self.summary = pickle.load(
            open('%s/data.pickle' % self.data_path, 'rb'))

        self.kfold = args.kfold
        self.train_pts = 'train_{}'.format(self.kfold)
        self.valid_pts = 'valid_{}'.format(self.kfold)
        self.test_pts = 'test'

        # We need to extract a list from a list for test patients
        self.summary[self.test_pts] = self.summary[self.test_pts][0]

        # Limit number of patients from a kfold (testing)
        if args.maxp:
            self.summary[self.train_pts] = self.summary[self.train_pts][0:args.maxp]
            self.summary[self.valid_pts] = self.summary[self.valid_pts][0:args.maxp]
            self.summary[self.test_pts] = self.summary[self.test_pts][0:args.maxp]

        # Image parameters
        self.image_size = args.image_size
        self.patch_size = args.patch_size

        # Training parameters
        self.lr = args.lr
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        self.batch_size = args.batch_size

        # Variable to scale the epoch step to the number of training patients used
        n_batches = len(self.summary['train']) if 'train' in self.summary else len(
            self.summary[self.train_pts])

        self.epoch_step = n_batches * \
            (self.image_size//self.patch_size)//self.batch_size  # integer value
        self.epoch = args.epoch
        self.initial_epoch = args.initial_epoch

        # Choose network version
        self.version = args.version

        # Generate model name from parameters
        self.model_outname = str(args.model_name) + "_bz" + str(self.batch_size) + \
            "_lr" + str(self.lr) + "_k" + str(self.kfold)

        # Resume previous training
        self.continue_train = args.continue_train

    def mkdir_(self, output):
        if not os.path.exists(output):
            os.makedirs(output)

    # Load numpy arrays
    def load_data(self, mode):
        args = self.args
        loader = data.DCMDataLoader(args, self.summary, mode)
        return loader.load_train_data(mode)

    # Data Generator for tensorfow
    def yield_data(self, stack):
        import tensorflow as tf

        for value in stack.values():
            # Check each patient has expected states
            for state, pair in value.items():
                if state == 'UNKNOWN':
                    print('Patient state for selected pair is unknown, skipping...')
                    continue
                # Extract file pair
                (ld_dict, hd_dict) = pair
                ld = ld_dict['numpy']
                hd = hd_dict['numpy']
                x = ld[-1, :, :, :]
                y = hd[-1, :, :, :]
                x = tf.cast(x, tf.float64)
                y = tf.cast(y, tf.float64)
                yield x, y

    def generator_train(self):
        stack = self.load_data(self.train_pts)
        return self.yield_data(stack)

    def generator_validate(self):
        stack = self.load_data(self.valid_pts)
        return self.yield_data(stack)

    # Return model name
    def get_model(self):
        # Last epoch step
        if os.path.exists(f'{self.model_outname}.h5'):
            model_name = self.model_outname + '.h5'
        else:
            # Last saved iteration
            checkpoint_models = glob.glob(
                'checkpoint/{}*.h5'.format(self.model_outname))
            if not checkpoint_models:
                print('No pretrained models found')
                exit(-1)
            model_name = checkpoint_models[-1]

        # Resume/load from specific epoch
        if self.initial_epoch > 0:
            model_name = f'checkpoint/{self.model_outname}_e{self.initial_epoch:03d}.h5'
            print("Resuming from model: {}".format(model_name))

        print(f'!LOADING MODEL: {model_name}!')
        return model_name

    def lr_step_decay(self, epoch, lr):
        drop_rate = 0.05
        epochs_drop = 1.0
        return self.lr*math.pow*(drop_rate, math.floor(epoch/epochs_drop))

    def decayed_learning_rate(self, epoch):
        return self.lr * 0.95 ** (epoch / 1000.0)

    def model_train(self, model_outname, x, y, z, d, epoch, epoch_step, batch_size,
                    lr, initial_epoch, verbose=1, train_pts=None, validate_pts=None,
                    initial_model=None, MULTIGPU=False, loss="mae"):

        import network, network_v2, network_v3
        import tensorflow as tf
        from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler

        # Generators
        data_train_gen = tf.data.Dataset.from_generator(self.generator_train,
                                                        output_types=(
                                                            tf.float64, tf.float64),
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 1)), tf.TensorShape((128, 128, 16, 1))))
        data_valid_gen = tf.data.Dataset.from_generator(self.generator_validate,
                                                        output_types=(
                                                            tf.float64, tf.float64),
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 1)), tf.TensorShape((128, 128, 16, 1))))
        # Make sure batches have the same outer dimension
        # repeat() to generate enough batches
        data_train_gen = data_train_gen.repeat().batch(batch_size, drop_remainder=True)
        data_valid_gen = data_valid_gen.repeat().batch(batch_size, drop_remainder=True)

        # Find pretrained model
        if self.continue_train:
            initial_model = self.get_model()

        # Load network version
        if self.version == 2:
            print('LOADING VERSON 2.0')
            self.model = network_v2.prepare_3D_unet(
                x, y, z, d, initialize_model=initial_model, lr=lr, loss=loss)
        elif self.version == 3:
            print('LOADING VERSON 3.0')
            self.model = network_v3.prepare_3D_unet(
                x, y, z, d, initialize_model=initial_model, lr=lr, loss=loss)
        elif self.version == 1:
            print('LOADING VERSON 1.0')
            self.model = network.prepare_3D_unet(
                x, y, z, d, initialize_model=initial_model, lr=lr, loss=loss)
        else:
            print('I do not know this model :(')

        self.mkdir_('checkpoint/TB/{}'.format(model_outname))
        filepath = os.path.join(
            'checkpoint', model_outname + "_e{epoch:03d}.h5")

        checkpoint = ModelCheckpoint(
            filepath, verbose=1, save_freq='epoch')  # use save_freq

        tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname),
                                 histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)

        stop_callback = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

        # lrs_callback = LearningRateScheduler(self.decayed_learning_rate, verbose = 1)

        callbacks_list = [checkpoint, tbCallBack, stop_callback]

        # Train model on dataset
        self.model.fit(data_train_gen,
                       steps_per_epoch=epoch_step,
                       validation_data=data_valid_gen,
                       validation_steps=100,
                       epochs=epoch,
                       verbose=1,
                       batch_size=batch_size,
                       callbacks=callbacks_list,
                       initial_epoch=initial_epoch)

        # Save model
        self.model.save(model_outname+".h5")
        print("Saved model to disk")

    # Test pretrained model

    def predict(self):
        stack = self.load_data(self.test_pts)
        from tensorflow.keras.models import load_model

        # Load pretrained model
        model_name = self.get_model()
        # model = load_model(model_name, compile=False)
        model = load_model(model_name)
        # Load test data
        stack = self.load_data(self.test_pts)
        for key, value in stack.items():
            for state, pair in value.items():
                if state == 'UNKNOWN':
                    print('Patient state for selected pair is unknown, skipping...')
                    continue

                (ld_dict, hd_dict) = pair
                ld = ld_dict['numpy']
                ld_raw = ld_dict['nifti']  # corresponding nifti file

                # print(type(ld))
                # print(type(ld_raw))

                img = ld_raw
                ld_data = ld.reshape(1, 128, 128, -1, 1)

                # Inference
                print(f'Predicting patient {key}...')
                z = self.patch_size

                predicted = np.empty((128, 128, 111))
                for z_index in range(int(z/2), 111-int(z/2)):
                    predicted_stack = model.predict(
                        ld_data[:, :, :, z_index-int(z/2):z_index+int(z/2), :].reshape(1, 128, 128, 16, 1))
                    # print(predicted_stack.shape)
                    if z_index == int(z/2):
                        for ind in range(int(z/2)):
                            predicted[:, :, ind] = predicted_stack[0,
                                                                   :, :, ind].reshape(128, 128)
                    if z_index == 111-int(z/2)-1:
                        for ind in range(int(z/2)):
                            predicted[:, :, z_index+ind] = predicted_stack[0,
                                                                           :, :, int(z/2)+ind].reshape(128, 128)
                    predicted[:, :, z_index] = predicted_stack[0,
                                                               :, :, int(z/2)].reshape(128, 128)
                predicted_full = predicted

                # Save NIFTI
                predicted_image = nib.Nifti1Image(
                    predicted_full, img.affine, img.header)
                save_dir = f'{self.data_path}/{self.model_outname}_predicted/{key}'
                self.mkdir_(save_dir)
                nib.save(predicted_image,
                         f'{save_dir}/{key}_{state.lower()}_predicted.nii.gz')
        print('Done.')

    def train(self):

        if os.path.exists(self.model_outname+".h5"):
            print("Model %s exists" % self.model_outname)

        # Initialise training
        self.model_train(self.model_outname, 128, 128, 16, 1, self.epoch,
                         self.epoch_step, self.batch_size, self.lr, self.initial_epoch)
