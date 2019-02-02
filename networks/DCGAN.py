from keras.layers import Input, Lambda
from keras.models import Model


def DCGAN(generator_model, discriminator_model, input_img_dim, patch_dim):
    generator_input = Input(shape=input_img_dim, name="DCGAN_input")

    generated_image = generator_model(generator_input)

    h, w = input_img_dim[1:]
    ph, pw = patch_dim

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(int(h / ph))]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(int(w / pw))]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1],
                                       col_idx[0]:col_idx[1]], output_shape=input_img_dim)(generated_image)
            list_gen_patch.append(x_patch)

    dcgan_output = discriminator_model(list_gen_patch)

    # dc_gan = Model(input=[generator_input], output=[generated_image, dcgan_output], name="DCGAN")
    dc_gan = Model(name="DCGAN", inputs=[generator_input], outputs=[generated_image, dcgan_output])
    return dc_gan
