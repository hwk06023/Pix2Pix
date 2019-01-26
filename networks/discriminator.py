from keras.layers import Flatten, Dense, Input, Reshape, Lambda, Concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np


def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
    stride = 2
    bn_mode = 2
    axis = 1
    input_layer = Input(shape=patch_dim)

    num_filters_start = 64
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn', mode=bn_mode, axis=axis)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_dim,
                                                      input_layer=input_layer,
                                                      nb_patches=nb_patches)
    return patch_gan_discriminator


def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):

    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    patch_gan = Model(input=[input_layer], output=[x, x_flat], name="patch_gan")

    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = Concatenate(x, mode="concat", name="merged_features")
    else:
        x = x[0]

    if len(x_mbd) > 1:
        x_mbd = Concatenate(x_mbd, mode="concat", name="merged_feature_mbd")
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    x_mbd = M(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = Concatenate([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')
    return discriminator


def lambda_output(input_shape):
    return input_shape[:2]


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x
