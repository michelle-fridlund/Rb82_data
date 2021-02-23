"""
Jan 19 12:59

michellef
##############################################################################

##############################################################################
"""

# import numpy as np
import data_generator as data
import matplotlib as mpl
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
        # pickle file
        self.summary = pickle.load(open('%s/data.pickle' % self.data_path, 'rb'))

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

        # Learning rate
        self.lr = args.lr

        self.batch_size = args.batch_size
        n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary[self.train_pts])
        self.epoch_step = n_batches*(args.image_size**2//args.patch_size**2)//self.batch_size
        self.epoch = args.epoch

        self.model_outname = str(args.model_name) + "_e" + \
            str(self.epoch) + "_bz" + str(self.batch_size) + "_lr" + \
            str(self.lr)+"_k" + str(self.kfold)

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

    def model_train(self, model_outname, x, y, z, d, epoch, epoch_step, batch_size,
                    lr, verbose=1, train_pts=None, validate_pts=None, initial_epoch=0,
                    initial_model=None, MULTIGPU=False, loss="mae"):

        import net_v1 as net
        import tensorflow as tf
        from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

        # Generators
        data_train_gen = tf.data.Dataset.from_generator(self.generator_train,
                                                        output_types=(
                                                            tf.float32, tf.float32),
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 2)), tf.TensorShape((128, 128, 16, 1))))
        data_valid_gen = tf.data.Dataset.from_generator(self.generator_validate,
                                                        output_types=(
                                                            tf.float32, tf.float32),
                                                        output_shapes=(tf.TensorShape((128, 128, 16, 2)), tf.TensorShape((128, 128, 16, 1))))

        data_valid_gen = data_valid_gen.repeat().batch(batch_size)
        data_train_gen = data_train_gen.repeat().batch(batch_size)

        self.model = net.prepare_3D_unet(x, y, z, d, initialize_model=initial_model, lr=lr, loss=loss)

        self.mkdir_('checkpoint/TB/{}'.format(model_outname))
        filepath = os.path.join('checkpoint', model_outname + "_e{epoch:03d}.h5")
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_freq='epoch')  # use save_freq

        tbCallBack = TensorBoard(log_dir='checkpoint/TB/{}'.format(model_outname),
                                 histogram_freq=0, write_graph=True, write_images=True, profile_batch=0)
        callbacks_list = [checkpoint, tbCallBack]

        # Train model on dataset
        self.model.fit(data_train_gen,
                       steps_per_epoch=epoch_step,
                       validation_data=data_valid_gen,
                       validation_steps=100,
                       epochs=epoch,
                       verbose=1,
                       callbacks=callbacks_list,
                       initial_epoch=initial_epoch)

        # Save model
        self.model.save(model_outname+".h5")
        print("Saved model to disk")

    def get_model(self):
        if os.path.exists(f'{self.model_outname}.h5'):
            model_name = self.model_outname + '.h5'
        else:
            checkpoint_models = glob.glob('checkpoint/{}*.h5'.format(self.model_outname))
            if not checkpoint_models:
                print('No pretrained models found')
                exit(-1)
            model_name = checkpoint_models[-1]
            return model_name

    def model_predict(self):
        #from tensorflow.keras.models import load_model
        # Load train data
        stack = self.load_data(self.test_pts)
        for key, value in stack.items():
            print(key)
            for state, pair in value.items():
                if state == 'UNKNOWN':
                    print('Patient state for selected pair is unknown, skipping...')
                    continue
                (ld, hd) = pair
        # #         x = ld[-1, :, :, :]

        # # #Load pretrained model
        # model_name = self.get_model
        # model = load_model(model_name)

        # predicted = np.empty((111, 128, 128))
        # x = 128
        # y = 128
        # z = 16
        # d = 2

        # for z_index in range(int(z/2), 111-int(z/2)):
        #     predicted_stack = model.predict(x[:, :, z_index-int(z/2):z_index+int(z/2), :].reshape(1, x, y, z, d))
        #     if z_index == int(z/2):
        #         for ind in range(int(z/2)):
        #             predicted[ind, :, :] = predicted_stack[0, :, :, ind].reshape(128, 128)
        #     if z_index == 111-int(z/2)-1:
        #         for ind in range(int(z/2)):
        #             predicted[z_index+ind, :, :] = predicted_stack[0, :, :, int(z/2)+ind].reshape(128, 128)
        #     predicted[z_index, :, :] = predicted_stack[0, :, :, int(z/2)].reshape(128, 128)
        # predicted_full = predicted
        # predicted_full += np.swapaxes(np.swapaxes(x[:, :, :, 0], 2, 1), 1, 0)

    # def train(self,args,LOG=None,MULTIGPU=False):
    def train(self):

        if os.path.exists(self.model_outname+".h5"):
            print("Model %s exists" % self.model_outname)

        # Initialise training
        self.model_train(self.model_outname, 128, 128, 16, 2, self.epoch, self.epoch_step, self.batch_size, self.lr)

    # TODO: Implement resume training / transfer learning
    # def test(self, args, x=args.image_size,y=dims_inplane,z=stack_of_slices,d=2,LOG=None):
        # model_predict(model,model_name,x,y,z,d,LOG=LOG,orientation='axial')
