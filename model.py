import numpy as np
import os
import keras

from keras.optimizers import Adam
from utils.facades_generator import facades_generator
from networks.generator import UNETGenerator
from networks.discriminator import PatchGanDiscriminator
from networks.DCGAN import DCGAN
from utils import patch_utils
from utils import logger
import time

input_channels = 1
output_channels = 1

input_img_dim = (input_channels, 256, 256)
#input_img_dim = (256, 256, input_channels)
output_img_dim = (output_channels, 256, 256)

sub_patch_dim = (256, 256)
nb_patch_patches, patch_gan_dim = patch_utils.num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

generator_nn = UNETGenerator(input_img_dim=input_img_dim, num_output_channels=output_channels)
generator_nn.summary()

discriminator_nn = PatchGanDiscriminator(output_img_dim=output_img_dim,
                                         patch_dim=patch_gan_dim, nb_patches=nb_patch_patches)
discriminator_nn.summary()

discriminator_nn.trainable = False

opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

generator_nn.compile(loss='mae', optimizer=opt_discriminator)

dc_gan_nn = DCGAN(generator_model=generator_nn,
                  discriminator_model=discriminator_nn,
                  input_img_dim=input_img_dim,
                  patch_dim=sub_patch_dim)

dc_gan_nn.summary()

loss = ['mae', 'binary_crossentropy']
loss_weights = [1E2, 1]
dc_gan_nn.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

discriminator_nn.trainable = True
discriminator_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

batch_size = 1
nb_epoch = 100
n_images_per_epoch = 400

print('Training starting...')
for epoch in range(0, nb_epoch):

    print('Epoch {}'.format(epoch))
    batch_counter = 1
    start = time.time()
    progbar = keras.utils.Progbar(n_images_per_epoch)

    tng_gen = facades_generator(data_dir_name='DATA', data_type='train', im_width=256, batch_size=batch_size)
    val_gen = facades_generator(data_dir_name='DATA', data_type='val', im_width=256, batch_size=batch_size)

    for mini_batch_i in range(0, n_images_per_epoch, batch_size):

        X_train_decoded_imgs, X_train_original_imgs = next(tng_gen)
        X_val_decoded_imgs, X_val_original_imgs = next(val_gen)

        X_discriminator, y_discriminator = patch_utils.get_disc_batch(X_train_original_imgs,
                                                                      X_train_decoded_imgs,
                                                                      generator_nn,
                                                                      batch_counter,
                                                                      patch_dim = sub_patch_dim)

        disc_loss = discriminator_nn.train_on_batch(X_discriminator, y_discriminator)

        X_gen_target, X_gen = next(patch_utils.gen_batch(X_train_original_imgs, X_train_decoded_imgs, batch_size))
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1

        discriminator_nn.trainable = False

        gen_loss = dc_gan_nn.train_on_batch(X_gen, [X_gen_target, y_gen])

        discriminator_nn.trainable = True

        batch_counter += 1

        D_log_loss = disc_loss
        gen_total_loss = gen_loss[0].tolist()
        gen_total_loss = min(gen_total_loss, 1000000)
        gen_mae = gen_loss[1].tolist()
        gen_mae = min(gen_mae, 1000000)
        gen_log_loss = gen_loss[2].tolist()
        gen_log_loss = min(gen_log_loss, 1000000)

        progbar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                        ("Gen total", gen_total_loss),
                                        ("Gen L1 (mae)", gen_mae),
                                        ("Gen logloss", gen_log_loss)])

        if batch_counter % 2 == 0:
            logger.plot_generated_batch(X_train_original_imgs, X_train_decoded_imgs, generator_nn, epoch, 'tng', mini_batch_i)

            X_full_val_batch, X_sketch_val_batch = next(patch_utils.gen_batch(X_val_original_imgs, X_val_decoded_imgs, batch_size))
            logger.plot_generated_batch(X_full_val_batch, X_sketch_val_batch, generator_nn, epoch, 'val', mini_batch_i)

        print(mini_batch_i)


    print('Epoch %s/%s, Time: %s\n' % (epoch + 1, nb_epoch, time.time() - start))

    if epoch % 2 == 0:
        gen_weights_path = os.path.join('./pix2pix_out/weights/gen_weights_epoch_%s.h5' % epoch)
        generator_nn.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join('./pix2pix_out/weights/disc_weights_epoch_%s.h5' % epoch)
        discriminator_nn.save_weights(disc_weights_path, overwrite=True)

        DCGAN_weights_path = os.path.join('./pix2pix_out/weights/DCGAN_weights_epoch_%s.h5' % epoch)
        dc_gan_nn.save_weights(DCGAN_weights_path, overwrite=True)