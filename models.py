from __future__ import absolute_import, division
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Cropping2D
from keras import layers
from keras.models import Model, Input
import keras.backend as K
from loss_function import kl_divergence, correlation_coefficient, nss, dre_loss
from keras.optimizers import Adam
from UpSampling2D_keras224 import UpSampling2D
import os

def DCN(input_tensor=None):
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv_1
    x = Conv2D(64, [3, 3], activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(img_input)
    x = Conv2D(64, [3, 3], activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)

    # conv_2
    x = Conv2D(128, [3, 3], activation='relu', padding='same', name='block2_conv1', data_format='channels_first')(x)
    x = Conv2D(128, [3, 3], activation='relu', padding='same', name='block2_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first')(x)

    # conv_3
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv1', data_format='channels_first')(x)
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv2', data_format='channels_first')(x)
    x = Conv2D(256, [3, 3], activation='relu', padding='same', name='block3_conv3', data_format='channels_first')(x)
    x_conv3_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same', data_format='channels_first')(x)

    # conv_4
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv1', data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv2', data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block4_conv3', data_format='channels_first')(x)
    x_conv4_3 = x
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same', data_format='channels_first')(x)

    # conv_5
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv1', dilation_rate=(2, 2), data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv2', dilation_rate=(2, 2), data_format='channels_first')(x)
    x = Conv2D(512, [3, 3], activation='relu', padding='same', name='block5_conv3', dilation_rate=(2, 2), data_format='channels_first')(x)

    # Create model
    model = Model(img_input, [x_conv3_3, x_conv4_3, x])

    # Load imagenet pretrained weights
    vgg16_weights = 'baseline_weights/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
    if os.path.exists(vgg16_weights):
        model.load_weights(vgg16_weights)

    return model


def SA_CFIM(x, data_format='channels_last'):

    # fine branch
    xf = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same', use_bias=False, data_format=data_format)(x)

    xf_p2 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(2, 2), use_bias=False, data_format=data_format)(xf)
    xf_p4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(4, 4), use_bias=False, data_format=data_format)(xf)
    xf_p8 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(8, 8), use_bias=False, data_format=data_format)(xf)

    xf = layers.add([xf_p2, xf_p4, xf_p8])

    # coarse branch
    xc = AveragePooling2D((2, 2), strides=(2, 2), padding='same', data_format=data_format)(x)
    xc = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same', use_bias=False, data_format=data_format)(xc)

    xc_p2 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(2, 2), use_bias=False, data_format=data_format)(xc)
    xc_p4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(4, 4), use_bias=False, data_format=data_format)(xc)
    xc_p8 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', dilation_rate=(8, 8), use_bias=False, data_format=data_format)(xc)

    xc = layers.add([xc_p2, xc_p4, xc_p8])
    xc = UpSampling2D(size=(2, 2), data_format=data_format, interpolation='bilinear')(xc)

    # x = layers.add([xc, xf_p2, xf_p4, xf_p8])
    x = layers.add([xf, xc])

    return x


def upsample_block(x, output_channels=None):
    # upsample the feature maps via deconv

    h_0 = x._keras_shape[1]
    w_0 = x._keras_shape[2]
    if not output_channels:
        output_channels = x._keras_shape[3]//2

    x_deconv = Conv2DTranspose(output_channels, kernel_size = [4, 4], strides=2, activation='relu',data_format='channels_last')(x)

    h_1 = x_deconv._keras_shape[1]
    w_1 = x_deconv._keras_shape[2]
    dh5 = int((h_1-h_0*2)/2)
    dw5 = int((w_1-w_0*2)/2)
    x_deconv_cropped = Cropping2D((dh5,dw5), data_format='channels_last')(x_deconv)

    return x_deconv_cropped

def RIM(x):
    c_0 = x._keras_shape[3]

    x = Conv2D(c_0, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False)(x)
    x = Conv2D(c_0, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False)(x)
    final_output = Conv2D(1, kernel_size=(3, 3), activation='relu',padding='same',use_bias=False)(x)

    return final_output


def ASD_SA(img_rows=480, img_cols=640, DRE_Loss=False, learning_rate=1e-5):
    inputimage = Input(shape=(3, img_rows, img_cols))
    base_model = DCN(input_tensor=inputimage)  #
    [F3, F4, F5] = base_model.output
    F3 = layers.Permute((2, 3, 1))(F3)  # 256
    F4 = layers.Permute((2, 3, 1))(F4)  # 512
    F5 = layers.Permute((2, 3, 1))(F5)  # 512

    # Scale-adaptive coarse and fine inception module
    F6 = SA_CFIM(F5)

    # path1
    F6_up1 = upsample_block(F6)
    AM6 = RIM(F6_up1)
    P6 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear')(AM6)

    # path2
    F5_up1 = upsample_block(F5)
    F5_up1 = layers.concatenate([F5_up1, F6_up1])
    F5_up2 = upsample_block(F5_up1)
    AM5 = RIM(F5_up2)
    P5 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')(AM5)

    # path3
    F4_up1 = upsample_block(F4)
    F4_up2 = upsample_block(F4_up1)
    F4_up2 = layers.concatenate([F4_up2, F5_up2])
    F4_up3 = upsample_block(F4_up2)
    AM4 = RIM(F4_up3)
    P4 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(AM4)

    # path4
    F3_up1 = upsample_block(F3)
    F3_up2 = upsample_block(F3_up1)
    F3_up2 = layers.concatenate([F3_up2, F4_up3])
    F3_up3 = upsample_block(F3_up2)
    AM3 = RIM(F3_up3)
    P3 = AM3

    # integrate the four attention maps to saliency map via convolution
    x_3456 = layers.concatenate([AM3, P4, P5, P6])
    final_saliency_map = Conv2D(1, (1, 1), activation='relu', padding='same', use_bias=False)(x_3456)

    # learning_rate = 1e-5
    # learning_rate = 1e-6
    # learning_rate = 5e-6
    if not DRE_Loss:
        model = Model(inputs=[inputimage], outputs=[P3, P3, P3,
                                                    P4, P4, P4,
                                                    P5, P5, P5,
                                                    P6, P6, P6,
                                                    final_saliency_map, final_saliency_map, final_saliency_map])
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss=[kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss],
                      loss_weights=[0.041, 0.041, 0.041,
                                    0.041, 0.041, 0.041,
                                    0.041, 0.041, 0.041,
                                    0.041, 0.041, 0.041,
                                    0.169, 0.169, 0.170 ])
    else:
        model = Model(inputs=[inputimage], outputs=[P3, P3, P3,
                                                    P4, P4, P4,
                                                    P5, P5, P5,
                                                    P6, P6, P6,
                                                    final_saliency_map, final_saliency_map, final_saliency_map,
                                                    final_saliency_map])
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss=[kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            kl_divergence, correlation_coefficient, nss,
                            dre_loss],
                            loss_weights=[0.035, 0.035, 0.035,
                                          0.035, 0.035, 0.035,
                                          0.035, 0.035, 0.035,
                                          0.035, 0.035, 0.035,
                                          0.145, 0.145, 0.145, 0.145 ])

    return model
