"""
FROM https://github.com/drivendataorg/concept-to-clinic/blob/master/prediction/src/preprocess/generators.py which is
Based on the
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
Fairly basic set of tools for real-time data augmentation on the volumetric
data. Extended for 3D objects augmentation.
"""

import keras.backend as K
import numpy as np
import scipy.ndimage
from keras.utils.data_utils import Sequence
from scipy import linalg
from six.moves import range

def random_rotation(x, rgs, channel_axis=0,
                    fill_mode='nearest', cval=0., order=0):
    """
    Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 4D.
        rg: Rotation range, in degrees.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    rgs = scipy.ndimage._ni_support._normalize_sequence(rgs, 3)
    theta = [np.random.uniform(-rg, rg) * np.pi / 180. for rg in rgs]

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, np.cos(theta[0]), -np.sin(theta[0]), 0],
                                  [0, np.sin(theta[0]), np.cos(theta[0]), 0],
                                  [0, 0, 0, 1]])

    rotation_matrix_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1]), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(theta[1]), 0, np.cos(theta[1]), 0],
                                  [0, 0, 0, 1]])

    rotation_matrix_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0, 0],
                                  [np.sin(theta[2]), np.cos(theta[2]), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    transform_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    axes = [i for i in range(len(x.shape)) if i != channel_axis]
    sides = [x.shape[i] for i in axes]
    transform_matrix = transform_matrix_offset_center(transform_matrix, sides)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval, order)
    return x


def random_shift(x, rgs, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """
    Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 4D.
        rgs: shift range, as a float fraction of the size.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    rgs = scipy.ndimage._ni_support._normalize_sequence(rgs, 3)
    axes = [i for i in range(4) if i != channel_axis]
    sides = [x.shape[i] for i in axes]
    translations = [np.random.uniform(-rg, rg) * side for rg, side in zip(rgs, sides)]
    #translations = [rg * side for rg, side in zip(rgs, sides)]
    translation_matrix = np.array([[1, 0, 0, translations[0]],
                                   [0, 1, 0, translations[1]],
                                   [0, 0, 1, translations[2]],
                                   [0, 0, 0, 1]])

    x = apply_transform(x, translation_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """
    Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 4D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    rgs = scipy.ndimage._ni_support._normalize_sequence(intensity, 3)
    shear = [np.random.uniform(-rg, rg) * np.pi / 180 for rg in rgs]
    shear_matrix = np.array([[1, -np.sin(shear[0]), np.cos(shear[1]), 0],
                             [np.cos(shear[0]), 1, -np.sin(shear[2]), 0],
                             [-np.sin(shear[1]), np.cos(shear[2]), 1, 0],
                             [0, 0, 0, 1]])

    axes = [i for i in range(4) if i != channel_axis]
    sides = [x.shape[i] for i in axes]
    transform_matrix = transform_matrix_offset_center(shear_matrix, sides)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_lower, zoom_upper, independent, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """
    Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 4D.
        zoom_lower: Float or Tuple of floats; zoom range lower bound.
            If scalar, then the same lower bound value will be set
            for each axis.
        zoom_upper: Float or Tuple of floats; zoom range upper bound.
            If scalar, then the same upper bound value will be set
            for each axis.
        independent: Boolean, whether to zoom each axis independently
            or with the same convex-combination coefficient `fctr`, ranged
            from 0 up to 1, so thar  `fctr` * lower + (1 - `fctr`) * upper.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    axes = [i for i in range(4) if i != channel_axis]
    zoom_lower = scipy.ndimage._ni_support._normalize_sequence(zoom_lower, len(axes))
    zoom_upper = scipy.ndimage._ni_support._normalize_sequence(zoom_upper, len(axes))

    if independent:
        zoom_fctr = [np.random.uniform(l, u) for l, u in zip(zoom_lower, zoom_upper)]
    else:
        fctr = np.random.uniform(0, 1)
        zoom_fctr = [fctr * l + (1 - fctr) * u for l, u in zip(zoom_lower, zoom_upper)]

    zoom_fctr = [1. / zf for zf in zoom_fctr]
    zoom_matrix = np.array([[zoom_fctr[0], 0, 0, 0],
                            [0, zoom_fctr[1], 0, 0],
                            [0, 0, zoom_fctr[2], 0],
                            [0, 0, 0, 1]])

    sides = [x.shape[i] for i in axes]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, sides)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, sides):
    sides = [float(side) / 2 - 0.5 for side in sides]
    offset_matrix = np.array([[1, 0, 0, sides[0]], [0, 1, 0, sides[1]], [0, 0, 1, sides[2]], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -sides[0]], [0, 1, 0, -sides[1]], [0, 0, 1, -sides[2]], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return np.array(transform_matrix.tolist())


def apply_transform(x, transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0., order=0):
    """
    Apply the image transformation specified by a matrix.
    # Arguments
        x: 4D numpy array, single patch.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [scipy.ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def add_random_noise(x,channel_axis,noise_level):
    # make channel axis the first axis always
    x = np.asarray(x).swapaxes(channel_axis, 0)
    
    # do noise addition on each channel
    for n in range(0,x.shape[0]):
        x_temp = x[n,...]
        n_elements = np.prod(x_temp.shape)
        std_xtemp = np.std(x_temp[np.where(x_temp>0)])
        #print(np.abs(noise_level*std_xtemp))
        noise = np.random.normal(0,np.abs(noise_level*std_xtemp),n_elements)
        x[n,...] = x_temp+noise.reshape(x_temp.shape)
        
    # make channel axis back to the original axis
    x_out = x.swapaxes(0, channel_axis)
    return x_out
    
    

class DataAugmentation3D(object):
    """
    Generate minibatches of image data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180), if scalar, then the same for each axis.
        shift_range: fraction of total length of the axes, if scalar, then the same for each axis.
        shear_range: shear intensity (shear angle in radians).
        zoom_lower: Float or Tuple of floats; zoom range lower bound.
            If scalar, then the same lower bound value will be set
            for each axis.
        zoom_upper: Float or Tuple of floats; zoom range upper bound.
            If scalar, then the same upper bound value will be set
            for each axis.
        zoom_independent: Boolean, whether to zoom each axis independently
            or with the same convex-combination coefficient `fctr`, ranged
            from 0 up to 1, so thar  `fctr` * lower + (1 - `fctr`) * upper.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'reflect'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        flip_axes: whether to randomly flip images through the axis from flip_axes.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        X_add_noise: relative noiselevel factor*std(x)
        X_shift_voxel: shift x relative to y. Input specify the maxmum list of number of voxel to be shifted in each direction 
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 shift_range=0.,
                 shear_range=0.,
                 zoom_lower=0.,
                 zoom_upper=0.,
                 zoom_independent=True,
                 channel_shift_range=0.,
                 fill_mode='reflect',
                 cval=0.,
                 flip_axis=None,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 X_add_noise=0,
                 X_shift_voxel=0):

        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.zoom_lower = zoom_lower
        self.zoom_upper = zoom_upper
        self.zoom_independent = zoom_independent
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.flip_axis = flip_axis
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.X_add_noise = X_add_noise
        self.X_shift_voxel = X_shift_voxel

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if self.data_format is None:
            self.data_format = K.image_data_format()
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.batchsize_axis = 4
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.batchsize_axis = 1

        self.mean = None
        self.std = None
        self.principal_components = None

        self.axes = [i for i in range(4) if i != self.channel_axis - 1]
        if self.rotation_range:
            self.rotation_range = scipy.ndimage._ni_support._normalize_sequence(self.rotation_range, 3)
            
        if self.X_shift_voxel:
            self.X_shift_voxel = scipy.ndimage._ni_support._normalize_sequence(self.X_shift_voxel, 3)
        
        if self.shift_range:
            self.shift_range = scipy.ndimage._ni_support._normalize_sequence(self.shift_range, 3)

        if self.shear_range:
            self.shear_range = scipy.ndimage._ni_support._normalize_sequence(self.shear_range, 3)

        if self.zoom_lower and self.zoom_upper:
            self.zoom_lower = scipy.ndimage._ni_support._normalize_sequence(zoom_lower, len(self.axes))
            self.zoom_upper = scipy.ndimage._ni_support._normalize_sequence(zoom_upper, len(self.axes))

    def random_transform_sample(self, x, y, seed=None):  # noqa: C901
        """
        Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single patch (predictor).
            y: 4D tensor, single patch (gold standard).            
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        channel_axis = self.channel_axis - 1
        sides = [x.shape[i] for i in self.axes]

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = [np.random.uniform(-rg, rg) * np.pi / 180 for rg in self.rotation_range]
            #theta = [rg * np.pi / 180 for rg in self.rotation_range]
        else:
            theta = 0

        if self.shift_range:
            shift = [np.random.uniform(-rg, rg) * side for rg, side in zip(self.shift_range, sides)]
            #shift = [rg * side for rg, side in zip(self.shift_range, sides)]
        else:
            shift = 0

        if self.shear_range:
            shear = [np.random.uniform(-rg, rg) * np.pi / 180 for rg in self.shear_range]
            #shear = [rg * np.pi / 180 for rg in self.shear_range]          
        else:
            shear = 0

        if self.zoom_lower and self.zoom_upper:
            if self.zoom_independent:
                zoom_fctr = [np.random.uniform(l, u) for l, u in zip(self.zoom_lower, self.zoom_upper)]
                #zoom_fctr = [u for l, u in zip(self.zoom_lower, self.zoom_upper)]
            else:
                fctr = np.random.uniform(0, 1)
                zoom_fctr = [fctr * l + (1 - fctr) * u for l, u in zip(self.zoom_lower, self.zoom_upper)]
            zoom_fctr = [1. / zf for zf in zoom_fctr]
        else:
            zoom_fctr = 0

        transform_matrix = None
        if theta != 0:
            rotation_matrix_x = np.array([[1, 0, 0, 0],
                                          [0, np.cos(theta[0]), -np.sin(theta[0]), 0],
                                          [0, np.sin(theta[0]), np.cos(theta[0]), 0],
                                          [0, 0, 0, 1]])

            rotation_matrix_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1]), 0],
                                          [0, 1, 0, 0],
                                          [-np.sin(theta[1]), 0, np.cos(theta[1]), 0],
                                          [0, 0, 0, 1]])

            rotation_matrix_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0, 0],
                                          [np.sin(theta[2]), np.cos(theta[2]), 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
            transform_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))

        if shift != 0:
            shift_matrix = np.array([[1, 0, 0, shift[0]],
                                     [0, 1, 0, shift[1]],
                                     [0, 0, 1, shift[2]],
                                     [0, 0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
#            shear_matrix = np.array([[1, -np.sin(shear[0]), np.cos(shear[1]), 0],
#                                     [np.cos(shear[0]), 1, -np.sin(shear[2]), 0],
#                                     [-np.sin(shear[1]), np.cos(shear[2]), 1, 0],
#                                     [0, 0, 0, 1]])
            shear_matrix = np.array([[1, np.tan(shear[0]), np.tan(shear[1]), 0],
                                     [np.tan(shear[0]), 1, np.tan(shear[2]), 0],
                                     [np.tan(shear[1]), np.tan(shear[2]), 1, 0],
                                     [0, 0, 0, 1]])        
            
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zoom_fctr != 0:
            zoom_matrix = np.array([[zoom_fctr[0], 0, 0, 0],
                                    [0, zoom_fctr[1], 0, 0],
                                    [0, 0, zoom_fctr[2], 0],
                                    [0, 0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:       
            axes = [i for i in range(4) if i != channel_axis]
            sides = [x.shape[i] for i in axes]
            #print(transform_matrix)             
            transform_matrix = transform_matrix_offset_center(transform_matrix, sides)
            #print(transform_matrix)     
            x = apply_transform(x, transform_matrix, channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            y = apply_transform(y, transform_matrix, channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        
#        if self.channel_shift_range != 0:
#            x = random_channel_shift(x, self.channel_shift_range, channel_axis)
            
        if self.flip_axis is not None:
            for axis in self.flip_axis:
                if np.random.random() < .5:
                    x = flip_axis(x, axis)
                    y = flip_axis(y, axis)
       
        if self.X_shift_voxel != 0:
            #n_voxels = self.shift_voxels_x
#            axes = [i for i in range(4) if i != channel_axis]
#            sides = [x.shape[i] for i in axes]
#            rg = np.array(n_voxels)*1/np.array(sides)
#            xtmp = x.copy()
#            x = random_shift(xtmp, rg, channel_axis=channel_axis, fill_mode=self.fill_mode, cval=self.cval)
            rgs = self.X_shift_voxel
            translations = [np.random.uniform(-rg, rg) for rg  in rgs]            
            translation_matrix = np.array([[1, 0, 0, translations[0]],
                                   [0, 1, 0, translations[1]],
                                   [0, 0, 1, translations[2]],
                                   [0, 0, 0, 1]])
            xtmp = x.copy()
            x = apply_transform(xtmp, translation_matrix, channel_axis, self.fill_mode, self.cval)
    
    
        # Noise is only applyed to X
        if self.X_add_noise != 0:
            noise_level = np.random.normal(0, self.X_add_noise)
            xtemp = x.copy()
            x = add_random_noise(xtemp, channel_axis, noise_level) # input is overridden if I dont use .copy()



        return x, y
    
    def random_transform_batch(self, x, y, seed=None):  # noqa: C901
        """
        Randomly augment a single image tensor.
        # Arguments
            x: 5D tensor, a batch of patches (predictor).
            y: 5D tensor, a batch of patches (gold standard).            
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
            
        """
        x_out = np.empty(x.shape)
        y_out = np.empty(y.shape)   
        
        
        for n in range(0,x.shape[self.batchsize_axis-1]):
            if self.data_format == 'channels_last': 
                x_out[n,...],y_out[n,...] = self.random_transform_sample(x[n,...],y[n,...])
            elif self.data_format == 'channels_first': 
                x_out[...,n],y_out[...,n] = self.random_transform_sample(x[...,n],y[...,n])
            else:
                print('NOT GOOD!')
                return -1
        return x_out, y_out
        
            
            