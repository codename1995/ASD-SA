from __future__ import division
import keras.backend as K

eps=2.2204e-16

# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    shape_r_out = y_pred._keras_shape[1]
    shape_c_out = y_pred._keras_shape[2]

    min_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.min(K.min(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    y_pred = y_pred - min_y_pred

    max_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.max(K.max(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.sum(K.sum(y_true, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.sum(K.sum(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)

    y_true /= (sum_y_true + eps)
    y_pred /= (sum_y_pred + eps)

    return K.sum(K.sum(y_true * K.log((y_true / (y_pred + eps)) + eps), axis=2), axis=1)

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    shape_r_out = y_pred._keras_shape[1]
    shape_c_out = y_pred._keras_shape[2]

    min_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.min(K.min(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    y_pred = y_pred - min_y_pred

    max_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.max(K.max(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.sum(K.sum(y_true, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.sum(K.sum(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)

    y_true /= (sum_y_true + eps)
    y_pred /= (sum_y_pred + eps)

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=1)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=1)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=1)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=1)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=1)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -2 * num / den

# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    shape_r_out = y_pred._keras_shape[1]
    shape_c_out = y_pred._keras_shape[2]

    max_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.max(K.max(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)

    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(
             K.repeat_elements(K.expand_dims(
                 K.expand_dims(y_mean),
                 axis=1), shape_r_out, axis=1),
                 axis=2), shape_c_out, axis=2)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(
             K.repeat_elements(K.expand_dims(
                 K.expand_dims(y_std),
                 axis=1), shape_r_out, axis=1),
                 axis=2), shape_c_out, axis=2)

    y_pred = (y_pred - y_mean) / (y_std + eps)

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=1) / K.sum(K.sum(y_true, axis=2), axis=1))

# Discriminative region enhancement loss
def dre_loss(y_true, y_pred):
    shape_r_out = y_pred._keras_shape[1]
    shape_c_out = y_pred._keras_shape[2]

    # Min-Max Normalization
    min_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.min(K.min(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    y_pred = y_pred - min_y_pred
    max_y_pred = K.repeat_elements(K.expand_dims(
                 K.repeat_elements(K.expand_dims(
                     K.max(K.max(y_pred, axis=2), axis=1)
                     , axis=1), shape_r_out, axis=1)
                     , axis=2), shape_c_out, axis=2)
    y_pred /= (max_y_pred + eps)

    return K.sum(K.sum(y_true * K.abs(y_pred-y_true), axis=2), axis=1) / K.sum(K.sum(y_true, axis=2), axis=1)
