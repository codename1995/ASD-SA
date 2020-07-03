# -*- coding: utf-8 -*-
"""Convolutional layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.engine.base_layer import Layer, InputSpec
from keras.legacy import interfaces
from keras.utils import conv_utils

import tensorflow as tf
import numpy as np

def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.
    # Arguments
        x: Tensor or variable.
    # Returns
        A tuple of integers (or None entries).
    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    {{np_implementation}}
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.
    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.
    # Returns
        A tensor.
    """
    return tf.transpose(x, perm=pattern)


def resize_images(x,
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
    """Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.
        interpolation: A string, one of `nearest` or `bilinear`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is
        neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        rows, cols = 2, 3
    else:
        rows, cols = 1, 2

    original_shape = int_shape(x)
    new_shape = tf.shape(x)[rows:cols + 1]
    new_shape *= tf.constant(np.array([height_factor, width_factor], dtype='int32'))

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf.image.resize_nearest_neighbor(x, new_shape)
    elif interpolation == 'bilinear':
        x = tf.image.resize_bilinear(x, new_shape)
    else:
        raise ValueError('interpolation should be one '
                         'of "nearest" or "bilinear".')
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])

    if original_shape[rows] is None:
        new_height = None
    else:
        new_height = original_shape[rows] * height_factor

    if original_shape[cols] is None:
        new_width = None
    else:
        new_width = original_shape[cols] * width_factor

    output_shape = (None, new_height, new_width, None)
    x.set_shape(transpose_shape(output_shape, data_format, spatial_axes=(1, 2)))
    return x

def transpose_shape(shape, target_format, spatial_axes):
    """Converts a tuple or a list to the correct `data_format`.
    It does so by switching the positions of its elements.
    # Arguments
        shape: Tuple or list, often representing shape,
            corresponding to `'channels_last'`.
        target_format: A string, either `'channels_first'` or `'channels_last'`.
        spatial_axes: A tuple of integers.
            Correspond to the indexes of the spatial axes.
            For example, if you pass a shape
            representing (batch_size, timesteps, rows, cols, channels),
            then `spatial_axes=(2, 3)`.
    # Returns
        A tuple or list, with the elements permuted according
        to `target_format`.
    # Example
    ```python
        >>> from keras.utils.generic_utils import transpose_shape
        >>> transpose_shape((16, 128, 128, 32),'channels_first', spatial_axes=(1, 2))
        (16, 32, 128, 128)
        >>> transpose_shape((16, 128, 128, 32), 'channels_last', spatial_axes=(1, 2))
        (16, 128, 128, 32)
        >>> transpose_shape((128, 128, 32), 'channels_first', spatial_axes=(0, 1))
        (32, 128, 128)
    ```
    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if target_format == 'channels_first':
        new_values = shape[:spatial_axes[0]]
        new_values += (shape[-1],)
        new_values += tuple(shape[x] for x in spatial_axes)

        if isinstance(shape, list):
            return list(new_values)
        return new_values
    elif target_format == 'channels_last':
        return shape
    else:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(target_format))

class _UpSampling(Layer):
    """Abstract nD UpSampling layer (private, used as implementation base).
    # Arguments
        size: Tuple of ints.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """
    def __init__(self, size, data_format=None, **kwargs):
        # self.rank is 1 for UpSampling1D, 2 for UpSampling2D.
        self.rank = len(size)
        self.size = size
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        super(_UpSampling, self).__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        size_all_dims = (1,) + self.size + (1,)
        spatial_axes = list(range(1, 1 + self.rank))
        size_all_dims = transpose_shape(size_all_dims,
                                        self.data_format,
                                        spatial_axes)
        output_shape = list(input_shape)
        for dim in range(len(output_shape)):
            if output_shape[dim] is not None:
                output_shape[dim] *= size_all_dims[dim]
        return tuple(output_shape)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(_UpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpSampling1D(_UpSampling):
    """Upsampling layer for 1D inputs.
    Repeats each temporal step `size` times along the time axis.
    # Arguments
        size: integer. Upsampling factor.
    # Input shape
        3D tensor with shape: `(batch, steps, features)`.
    # Output shape
        3D tensor with shape: `(batch, upsampled_steps, features)`.
    """

    @interfaces.legacy_upsampling1d_support
    def __init__(self, size=2, **kwargs):
        super(UpSampling1D, self).__init__((int(size),), 'channels_last', **kwargs)

    def call(self, inputs):
        output = K.repeat_elements(inputs, self.size[0], axis=1)
        return output

    def get_config(self):
        config = super(UpSampling1D, self).get_config()
        config['size'] = self.size[0]
        config.pop('data_format')
        return config


class UpSampling2D(_UpSampling):
    """Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by size[0] and size[1] respectively.
    # Arguments
        size: int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation: A string, one of `nearest` or `bilinear`.
            Note that CNTK does not support yet the `bilinear` upscaling
            and that with Theano, only `size=(2, 2)` is possible.
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), data_format=None, interpolation='nearest',
                 **kwargs):
        normalized_size = conv_utils.normalize_tuple(size, 2, 'size')
        super(UpSampling2D, self).__init__(normalized_size, data_format, **kwargs)
        if interpolation not in ['nearest', 'bilinear']:
            raise ValueError('interpolation should be one '
                             'of "nearest" or "bilinear".')
        self.interpolation = interpolation

    def call(self, inputs):
        return resize_images(inputs, self.size[0], self.size[1],
                               self.data_format, self.interpolation)

    def get_config(self):
        config = super(UpSampling2D, self).get_config()
        config['interpolation'] = self.interpolation
        return config