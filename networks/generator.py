from keras.layers import Activation, Input, Dropout, Concatenate, Conv2D, UpSampling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K


def make_generator_ae(input_layer, num_output_filters):

    stride = 2
    filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]

    encoder = input_layer
    for filter_size in filter_sizes:
        encoder = Conv2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(encoder)
        if filter_size != 64:
            encoder = BatchNormalization()(encoder)
        encoder = Activation(LeakyReLU(alpha=0.2))(encoder)

    filter_sizes = [512, 512, 512, 512, 512, 256, 128, 64]

    decoder = encoder
    for filter_size in filter_sizes:
        decoder = UpSampling2D(size=(2, 2))(decoder)
        decoder = Conv2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(p=0.4)(decoder)
        decoder = Activation('relu')(decoder)

    decoder = Conv2D(nb_filter=num_output_filters, nb_row=4, nb_col=4, border_mode='same')(decoder)
    generator = Activation('tanh')(decoder)
    return generator


def UNETGenerator(input_img_dim, num_output_channels):

    K.set_image_data_format('channels_first')
    input_layer = Input(shape=input_img_dim, name="unet_input")

    en_1 = Conv2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(input_layer)
    en_1 = LeakyReLU(alpha=0.2)(en_1)

    en_2 = Conv2D(nb_filter=128, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_1)
    en_2 = BatchNormalization(axis=1)(en_2)
    en_2 = LeakyReLU(alpha=0.2)(en_2)

    en_3 = Conv2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_2)
    en_3 = BatchNormalization(axis=1)(en_3)
    en_3 = LeakyReLU(alpha=0.2)(en_3)

    en_4 = Conv2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_3)
    en_4 = BatchNormalization(axis=1)(en_4)
    en_4 = LeakyReLU(alpha=0.2)(en_4)

    en_5 = Conv2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_4)
    en_5 = BatchNormalization(axis=1)(en_5)
    en_5 = LeakyReLU(alpha=0.2)(en_5)

    en_6 = Conv2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_5)
    en_6 = BatchNormalization(axis=1)(en_6)
    en_6 = LeakyReLU(alpha=0.2)(en_6)

    en_7 = Conv2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(2, 2))(en_6)
    en_7 = BatchNormalization(axis=1)(en_7)
    en_7 = LeakyReLU(alpha=0.2)(en_7)

    de_1 = UpSampling2D(size=(2, 2))(en_7)
    de_1 = Conv2D(nb_filter=1024, nb_row=4, nb_col=4,strides= 2, border_mode='same')(de_1)
    de_1 = BatchNormalization(axis=1)(de_1)
    de_1 = Dropout(p=0.5)(de_1)
    de_1 = Concatenate(axis=1)([de_1, en_7])
    de_1 = Activation('relu')(de_1)
    print(de_1.get_shape())

    de_2 = UpSampling2D(size=(2, 2))(de_1)
    de_2 = Conv2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_2)
    de_2 = BatchNormalization(axis=1)(de_2)
    de_2 = Dropout(p=0.5)(de_2)
    de_2 = Concatenate(axis=1)([de_2, en_6])
    de_2 = Activation('relu')(de_2)

    de_3 = UpSampling2D(size=(2, 2))(de_2)
    de_3 = Conv2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_3)
    de_3 = BatchNormalization(axis=1)(de_3)
    de_3 = Dropout(p=0.5)(de_3)
    de_3 = Concatenate(axis=1)([de_3, en_5])
    de_3 = Activation('relu')(de_3)

    de_4 = UpSampling2D(size=(2, 2))(de_3)
    de_4 = Conv2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_4)
    de_4 = BatchNormalization(axis=1)(de_4)
    de_4 = Dropout(p=0.5)(de_4)
    de_4 = Concatenate(axis=1)([de_4, en_4])
    de_4 = Activation('relu')(de_4)

    de_5 = UpSampling2D(size=(2, 2))(de_4)
    de_5 = Conv2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_5)
    de_5 = BatchNormalization(axis=1)(de_5)
    de_5 = Dropout(p=0.5)(de_5)
    de_5 = Concatenate(axis=1)([de_5, en_3])
    de_5 = Activation('relu')(de_5)

    de_6 = UpSampling2D(size=(2, 2))(de_5)
    de_6 = Conv2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same')(de_6)
    de_6 = BatchNormalization(axis=1)(de_6)
    de_6 = Dropout(p=0.5)(de_6)
    de_6 = Concatenate(axis=1)([de_6, en_2])
    de_6 = Activation('relu')(de_6)

    de_7 = UpSampling2D(size=(2, 2))(de_6)
    de_7 = Conv2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same')(de_7)
    de_7 = BatchNormalization(axis=1)(de_7)
    de_7 = Dropout(p=0.5)(de_7)
    de_7 = Concatenate(axis=1)([de_7, en_1])
    de_7 = Activation('relu')(de_7)

    de_8 = UpSampling2D(size=(2, 2))(de_7)
    de_8 = Conv2D(nb_filter=num_output_channels, nb_row=4, nb_col=4, border_mode='same')(de_8)
    de_8 = Activation('tanh')(de_8)

    unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
    return unet_generator
