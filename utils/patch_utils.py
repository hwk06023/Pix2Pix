import numpy as np


def num_patches(output_img_dim=(3, 256, 256), sub_patch_dim=(64, 64)):
    nb_non_overlaping_patches = (output_img_dim[1] / sub_patch_dim[0]) * (output_img_dim[2] / sub_patch_dim[1])

    patch_disc_img_dim = (output_img_dim[0], sub_patch_dim[0], sub_patch_dim[1])

    return int(nb_non_overlaping_patches), patch_disc_img_dim


def extract_patches(images, sub_patch_dim):
    im_height, im_width = images.shape[2:]
    patch_height, patch_width = sub_patch_dim

    x_spots = range(0, im_width, patch_width)

    y_spots = range(0, im_height, patch_height)
    all_patches = []

    for y in y_spots:
        for x in x_spots:
            image_patches = images[:, :, y: y+patch_height, x: x+patch_width]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))
    return all_patches


def get_disc_batch(X_original_batch, X_decoded_batch, generator_model, batch_counter, patch_dim,
                   label_smoothing=False, label_flipping=0):

    if batch_counter % 2 == 0:
        X_disc = generator_model.predict(X_decoded_batch)

        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_original_batch

        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    X_disc = extract_patches(images=X_disc, sub_patch_dim=patch_dim)

    return X_disc, y_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2