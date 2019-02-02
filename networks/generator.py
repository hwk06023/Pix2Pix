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
        decoder = Dropout(rate=0.4)(decoder)
        decoder = Activation('relu')(decoder)

    decoder = Conv2D(nb_filter=num_output_filters, nb_row=4, nb_col=4, border_mode='same')(decoder)
    generator = Activation('tanh')(decoder)
    return generator


def UNETGenerator(input_img_dim, num_output_channels):

    K.set_image_data_format('channels_first')
    input_layer = Input(shape=input_img_dim, name="unet_input")

    en_1 = Conv2D(kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same")(input_layer)
    en_1 = LeakyReLU(alpha=0.2)(en_1)

    en_2 = Conv2D(kernel_size=(4, 4), filters=128, strides=(2, 2), padding="same")(en_1)
    en_2 = BatchNormalization(axis=1)(en_2)
    en_2 = LeakyReLU(alpha=0.2)(en_2)

    en_3 = Conv2D(kernel_size=(4, 4), filters=256, strides=(2, 2), padding="same")(en_2)
    en_3 = BatchNormalization(axis=1)(en_3)
    en_3 = LeakyReLU(alpha=0.2)(en_3)

    en_4 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_3)
    en_4 = BatchNormalization(axis=1)(en_4)
    en_4 = LeakyReLU(alpha=0.2)(en_4)

    en_5 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_4)
    en_5 = BatchNormalization(axis=1)(en_5)
    en_5 = LeakyReLU(alpha=0.2)(en_5)

    en_6 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_5)
    en_6 = BatchNormalization(axis=1)(en_6)
    en_6 = LeakyReLU(alpha=0.2)(en_6)

    en_7 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_6)
    en_7 = BatchNormalization(axis=1)(en_7)
    en_7 = LeakyReLU(alpha=0.2)(en_7)

    de_1 = UpSampling2D(size=(2, 2))(en_7)
    de_1 = Conv2D(strides=2, kernel_size=(4, 4), filters=1024, padding="same")(de_1)
    de_1 = BatchNormalization(axis=1)(de_1)
    de_1 = Dropout(rate=0.5)(de_1)
    de_1 = Concatenate(axis=1)([de_1, en_7])
    de_1 = Activation('relu')(de_1)

    de_2 = UpSampling2D(size=(2, 2))(de_1)
    de_2 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_2)
    de_2 = BatchNormalization(axis=1)(de_2)
    de_2 = Dropout(rate=0.5)(de_2)
    de_2 = Concatenate(axis=1)([de_2, en_6])
    de_2 = Activation('relu')(de_2)

    de_3 = UpSampling2D(size=(2, 2))(de_2)
    de_3 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_3)
    de_3 = BatchNormalization(axis=1)(de_3)
    de_3 = Dropout(rate=0.5)(de_3)
    de_3 = Concatenate(axis=1)([de_3, en_5])
    de_3 = Activation('relu')(de_3)

    de_4 = UpSampling2D(size=(2, 2))(de_3)
    de_4 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_4)
    de_4 = BatchNormalization(axis=1)(de_4)
    de_4 = Dropout(rate=0.5)(de_4)
    de_4 = Concatenate(axis=1)([de_4, en_4])
    de_4 = Activation('relu')(de_4)

    de_5 = UpSampling2D(size=(2, 2))(de_4)
    de_5 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_5)
    de_5 = BatchNormalization(axis=1)(de_5)
    de_5 = Dropout(rate=0.5)(de_5)
    de_5 = Concatenate(axis=1)([de_5, en_3])
    de_5 = Activation('relu')(de_5)

    de_6 = UpSampling2D(size=(2, 2))(de_5)
    de_6 =Conv2D(kernel_size=(4, 4), filters=512, padding="same")(de_6)
    de_6 = BatchNormalization(axis=1)(de_6)
    de_6 = Dropout(rate=0.5)(de_6)
    de_6 = Concatenate(axis=1)([de_6, en_2])
    de_6 = Activation('relu')(de_6)

    de_7 = UpSampling2D(size=(2, 2))(de_6)
    de_7 = Conv2D(kernel_size=(4, 4), filters=256, padding="same")(de_7)
    de_7 = BatchNormalization(axis=1)(de_7)
    de_7 = Dropout(rate=0.5)(de_7)
    de_7 = Concatenate(axis=1)([de_7, en_1])
    de_7 = Activation('relu')(de_7)

    de_8 = UpSampling2D(size=(2, 2))(de_7)
    de_8 = Conv2D(kernel_size=(4, 4), filters=1, padding="same")(de_8)
    de_8 = Activation('tanh')(de_8)

    #unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
    unet_generator = Model(name="unet_generator", inputs=[input_layer], outputs=[de_8])
    return unet_generator
