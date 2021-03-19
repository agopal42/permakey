import tensorflow as tf
import numpy as np
from skimage.transform import resize
import os
from functools import partial
from zipfile import ZipFile, BadZipFile
from utils import ENV_NAME_TO_GYM_NAME
AUTOTUNE = tf.data.experimental.AUTOTUNE


def add_noise(img, noise_type):
    """
    helper function to add desired noise type to img frame
    :param img: img either (h, w, 1) or (h, w, 3) numpy array
    :param noise_type: [vertical, horizontal, both] type of additive noise
    :return: noisy img
    """
    width = 5
    line_colour = np.array([128, 0, 255]) / 255.0
    if noise_type == "vertical":
        # pick random x-coordinate
        img_shape = img.shape
        x_loc = np.random.randint(0, img_shape[1]-width)
        line_pixel = np.tile(line_colour, (img_shape[0], width, 1))
        img[:, x_loc:x_loc+width, :] = line_pixel
    elif noise_type == "horizontal":
        # pick random y-coordinate
        img_shape = img.shape
        y_loc = np.random.randint(0, img_shape[0]-width)
        line_pixel = np.tile(line_colour, (width, img_shape[1], 1))
        img[y_loc:y_loc+width, :] = line_pixel
    elif noise_type == "both":
        # pick random x-y coordinate
        img_shape = img.shape
        x_loc = np.random.randint(0, img_shape[1] - width)
        col_pixel = np.tile(line_colour, (img_shape[0], width, 1))
        img[:, x_loc:x_loc + width, :] = col_pixel
        y_loc = np.random.randint(0, img_shape[0] - width)
        row_pixel = np.tile(line_colour, (width, img_shape[1], 1))
        img[y_loc:y_loc + width, :] = row_pixel
    return img


def prepare_dataset(ds, batch_size, loss_to_use, split, shuffle_buffer_size=1000):
    """
    This is a small dataset, only load it once, and keep it in memory.
    Use `.cache(filename)` to cache preprocessing work for datasets that don't
    fit in memory.
    :param ds: Tensorflow Dataset object
    :param batch_size: batch size
    :param loss_to_use: "transporter" or "pkey"
    :param split: data set split
    :param shuffle_buffer_size: Size of the buffer to use for shuffling
    :return:
    """

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)

    # `prefetch` lets the dataset fetch batches in the background while the model is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    if loss_to_use == "transporter" and split == "train":
        ds = ds.take(10 ** 5)  # train data size 100k
    elif loss_to_use == "transporter" and (split == "valid" or split == "test"):
        ds = ds.take(3*10**2)  # valid/test data size 1k
    return ds


def atari_generator_func(filenames_list, noise_type="vertical", rgb=False):
    """
    A python generator for Atari frames.
    :param filenames_list: list of data filenames containing observations.
    :param noise_type: type of noise to be added for "noisy" Atari exps
    :param rgb: (bool) If RGB (True) Grayscale (False)
    :return:
    """

    for file in filenames_list:
        try:
            with ZipFile(file, 'r') as zf:
                data = np.load(file)
                if not rgb:  # gray-scale
                    obs = data['observations']
                    for i in range(obs.shape[0]):
                        yield obs[i, :, :, 3][:, :, None]

                elif rgb:  # atari full-sized colored frames (160, 210, 3)
                    obs = data['frames'] / 255.0
                    for i in range(obs.shape[0]):
                        if noise_type == "none":
                            yield resize(obs[i], (84, 84), order=0)
                        elif noise_type != "none":
                            yield add_noise(resize(obs[i], (84, 84), order=0), noise_type)
        except BadZipFile:
            print("Corrupted zip file ignored..")


def transporter_atari_gen(filenames_list, noise_type, rgb=False):
    """
    :param filenames_list: list of data filenames containing atari frames.
    :param noise_type: type of noise to be added for "noisy" Atari exps
    :param rgb: (bool) (True) if rgb frame (False) for greyscale
    :return:
    """

    for file in filenames_list:
        try:
            with ZipFile(file, 'r') as zf:
                data = np.load(file)
                if not rgb:
                    obs = data['observations']
                    obs_shape = obs.shape
                    for count in range(int(10**5 / len(filenames_list))):
                        window_start_idx = np.random.randint(0, obs_shape[0]-100, 1)[0]
                        window_obs = obs[window_start_idx:window_start_idx+100, :, :, :]
                        t1 = np.random.randint(0, 49, 1)[0]
                        t2 = np.random.randint(50, 99, 1)[0]
                        image_a = window_obs[t1, :, :, 3]
                        image_b = window_obs[t2, :, :, 3]
                        image_a, image_b = image_a[:, :, None], image_b[:, :, None]
                        if noise_type == "none":
                            yield np.stack([image_a, image_b], axis=3)
                        elif noise_type != "none":
                            yield np.stack([add_noise(image_a, noise_type),
                                            add_noise(image_b, noise_type)], axis=3)

                elif rgb:
                    obs = data['frames'] / 255.0
                    obs_shape = obs.shape
                    for count in range(int(10**5 / len(filenames_list))):
                        window_start_idx = np.random.randint(0, obs_shape[0]-100)
                        window_obs = obs[window_start_idx:window_start_idx+100, :, :, :]
                        t1 = np.random.randint(0, 49, 1)[0]
                        t2 = np.random.randint(50, 99, 1)[0]
                        image_a = resize(window_obs[t1, :, :, :], (84, 84), order=0)
                        image_b = resize(window_obs[t2, :, :, :], (84, 84), order=0)
                        if noise_type == "none":
                            yield np.stack([image_a, image_b], axis=3)
                        elif noise_type != "none":
                            yield np.stack([add_noise(image_a, noise_type),
                                            add_noise(image_b, noise_type)], axis=3)
        except BadZipFile:
            print("Corrupted zip file ignored...")


def deepmind_atari(data_path, env_name, split, loss_to_use, batch_size,
                   noise_type, rgb_frames):
    """
    Create TF Dataset object from DM-style atari frames (84, 84, 1)
    :param data_path: location of root data dir.
    :param env_name: environment name.
    :param split: {'train', 'valid', 'test'}
    :param loss_to_use: "pkey" or "transporter" loss
    :param batch_size: batch size.
    :param noise_type: noise_type [vertical, horizontal, both, back_flicker, none] to be added
    :param rgb_frames: boolean for indicating rgb frames (True) or grayscale frames (False)
    :return: tf.data.Dataset object with DM-style atari frames
    """

    env_name = ENV_NAME_TO_GYM_NAME[env_name]
    if env_name is None:
        raise ValueError("Unsupported environment: %s" % env_name)

    if split == "train":
        data_dir = os.path.join(data_path, "train", env_name)
    elif split == "valid" or split == "test":
        data_dir = os.path.join(data_path, "test", env_name)
    else:
        raise ValueError("Unknown dataset split: %s" % split)

    assert os.path.isdir(data_dir), "%s does not exist" % data_dir

    # Load files
    filenames_list = []
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:  # opens each .npz file
            filenames_list.append(os.path.join(subdir, file))

    # splitting test folder files into 2 halves i.e. valid + test
    if split == "valid":
        filenames_list = filenames_list[0:round(len(filenames_list)/2)]
    elif split == "test":
        filenames_list = filenames_list[round(len(filenames_list)/2)+1:]

    # Different processing depending on loss
    if loss_to_use == "pkey":
        # creating training dataset class
        gen_func = partial(atari_generator_func, filenames_list=filenames_list,
                           noise_type=noise_type, rgb=rgb_frames)
        ds = tf.data.Dataset.from_generator(gen_func, output_types=tf.float32)
        ds = prepare_dataset(ds, batch_size, loss_to_use, split)

    elif loss_to_use == "transporter":
        # creating training dataset class
        train_gen_func = partial(transporter_atari_gen, filenames_list=filenames_list,
                                noise_type=noise_type, rgb=rgb_frames)
        ds = tf.data.Dataset.from_generator(train_gen_func, output_types=tf.float32)
        ds = prepare_dataset(ds, batch_size, loss_to_use, split)

    else:
        raise ValueError("Unknown loss to use: %s" % loss_to_use)
    return ds
