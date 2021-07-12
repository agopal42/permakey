import tensorflow as tf

l2_reg = tf.keras.regularizers.l2(l=0.1)


class TransporterEncoder(tf.keras.Model):

    def __init__(self):
        super(TransporterEncoder, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                            padding="same", kernel_regularizer=l2_reg)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                             padding="same", kernel_regularizer=l2_reg)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2,
                                             padding="same", kernel_regularizer=l2_reg)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.conv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                             padding="same", kernel_regularizer=l2_reg)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        self.filters = [16, 16, 32, 32]
        self.kernels = [3, 3, 3, 3]
        self.strides = [1, 1, 2, 1]

    def call(self, inputs, training=True):
        h1 = self.relu_1(self.bn_1(self.conv_1(inputs), training=training))
        h2 = self.relu_2(self.bn_2(self.conv_2(h1), training=training))
        h3 = self.relu_3(self.bn_3(self.conv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.conv_4(h3), training=training))
        return h4


class TransporterDecoder(tf.keras.Model):

    def __init__(self, colour):
        super(TransporterDecoder, self).__init__()

        self.deconv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                            padding="same", kernel_regularizer=l2_reg)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.deconv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                            padding="same", kernel_regularizer=l2_reg)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.deconv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                               padding="same", kernel_regularizer=l2_reg)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.deconv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                               padding="same", kernel_regularizer=l2_reg)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        if colour:
            self.deconv_5 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1,
                                            padding="same", kernel_regularizer=l2_reg)
            self.bn_5 = tf.keras.layers.BatchNormalization()

        elif not colour:
            self.deconv_5 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1,
                                            padding="same", kernel_regularizer=l2_reg)
            self.bn_5 = tf.keras.layers.BatchNormalization()

    def call(self, features, training=True):

        h1 = self.relu_1(self.bn_1(self.deconv_1(features), training=training))
        h2 = self.relu_2(self.bn_2(self.deconv_2(h1), training=training))
        # 2X upsample
        h2_shape = h2.shape
        h2 = tf.image.resize(h2, (2*h2_shape[1], 2*h2_shape[2]), method='bilinear')
        h3 = self.relu_3(self.bn_3(self.deconv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.deconv_4(h3), training=training))
        reconstructed_image = tf.keras.activations.sigmoid(self.bn_5(self.deconv_5(h4), training=training))
        return reconstructed_image


def get_keypoint_data_from_feature_map(feature_map, gauss_std):
    """Returns keypoint information from a feature map.
    Args:
      feature_map: [B, H, W, K] Tensor, should be activations from a convnet.
      gauss_std: float, the standard deviation of the gaussians to be put around
        the keypoints.
    Returns:
      a dict with keys:
        'centers': A tensor of shape [B, K, 2] of the center locations for each
            of the K keypoints.
        'heatmaps': A tensor of shape [B, H, W, K] of gaussian maps over the
            keypoints.
    """
    gauss_mu = _get_keypoint_mus(feature_map)
    map_size = feature_map.shape.as_list()[1:3]
    gauss_maps = _get_gaussian_maps(gauss_mu, map_size, 1.0 / gauss_std)

    return gauss_mu, gauss_maps


def _get_keypoint_mus(keypoint_features):
    """Returns the keypoint center points.
    Args:
      keypoint_features: A tensor of shape [B, F_h, F_w, K] where K is the number
        of keypoints to extract.
    Returns:
      A tensor of shape [B, K, 2] of the y, x center points of each keypoint. Each
        center point are in the range [-1, 1]^2. Note: the first element is the y
        coordinate, the second is the x coordinate.
    """
    gauss_y = _get_coord(keypoint_features, 1)
    gauss_x = _get_coord(keypoint_features, 2)
    gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)
    return gauss_mu


def _get_coord(features, axis):
    """Returns the keypoint coordinate encoding for the given axis.
    Args:
      features: A tensor of shape [B, F_h, F_w, K] where K is the number of
        keypoints to extract.
      axis: `int` which axis to extract the coordinate for. Has to be axis 1 or 2.
    Returns:
      A tensor of shape [B, K] containing the keypoint centers along the given
        axis. The location is given in the range [-1, 1].
    """
    if axis != 1 and axis != 2:
        raise ValueError("Axis needs to be 1 or 2.")

    other_axis = 1 if axis == 2 else 2
    axis_size = features.shape[axis]

    # Compute the normalized weight for each row/column along the axis
    g_c_prob = tf.reduce_mean(features, axis=other_axis)
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)

    # Linear combination of the interval [-1, 1] using the normalized weights to
    # give a single coordinate in the same interval [-1, 1]
    scale = tf.cast(tf.linspace(-1.0, 1.0, axis_size), tf.float32)
    scale = tf.reshape(scale, [1, axis_size, 1])
    coordinate = tf.reduce_sum(g_c_prob * scale, axis=1)
    return coordinate


def _get_gaussian_maps(mu, map_size, inv_std, power=2):
    """Transforms the keypoint center points to a gaussian masks."""
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = tf.cast(tf.linspace(-1.0, 1.0, map_size[0]), tf.float32)
    x = tf.cast(tf.linspace(-1.0, 1.0, map_size[1]), tf.float32)

    mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

    y = tf.reshape(y, [1, 1, map_size[0], 1])
    x = tf.reshape(x, [1, 1, 1, map_size[1]])

    g_y = tf.pow(y - mu_y, power)
    g_x = tf.pow(x - mu_x, power)
    dist = (g_y + g_x) * tf.pow(inv_std, power)
    g_yx = tf.exp(-dist)

    g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
    return g_yx


class TransporterKeypointer(tf.keras.Model):
    """Module for extracting keypoints from an image."""
    def __init__(self, num_keypoints, gauss_std):
        super(TransporterKeypointer, self).__init__()
        self.num_keypoints = num_keypoints
        self.gauss_std = gauss_std
        self.keypoint_encoder = TransporterEncoder()
        self.conv = tf.keras.layers.Conv2D(filters=self.num_keypoints, kernel_size=1,
                                           strides=1, kernel_regularizer=l2_reg)

    def call(self, image, training=True):
        image_features = self.keypoint_encoder(image, training=training)
        keypoint_features = self.conv(image_features)
        return get_keypoint_data_from_feature_map(keypoint_features, self.gauss_std)


class ConvEncoder(tf.keras.Model):

    def __init__(self, latent_dim_size):
        super(ConvEncoder, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=1,
                                             padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
                                             padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2,
                                             padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                             padding="same")
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(2*latent_dim_size)  # no activation fc_size = size(mu) + size(sigma)

        # useful vars
        self.filters = [32, 64, 64, 128]
        self.kernels = [4, 3, 3, 3]
        self.strides = [1, 2, 2, 1]

    def call(self, inputs, training=True):
        h1 = self.relu_1(self.bn_1(self.conv_1(inputs), training=training))
        h2 = self.relu_2(self.bn_2(self.conv_2(h1), training=training))
        h3 = self.relu_3(self.bn_3(self.conv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.conv_4(h3), training=training))
        flatten_h4 = self.flatten(h4)
        mean, var = tf.split(self.fc(flatten_h4), num_or_size_splits=2, axis=1)
        return mean, tf.nn.softplus(var), [h1, h2, h3, h4]  # returning +ve vars


class DeconvDecoder(tf.keras.Model):

    def __init__(self, colour):
        super(DeconvDecoder, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(15*15)
        self.relu_d = tf.keras.layers.ReLU()

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=1)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.deconv_2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.deconv_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2)
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.deconv_4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=1)
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        if colour:  # color image
            self.deconv_5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1)
            self.bn_5 = tf.keras.layers.BatchNormalization()
        elif not colour:  # greyscale image
            self.deconv_5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1)
            self.bn_5 = tf.keras.layers.BatchNormalization()

    def call(self, z, training=True):
        # flattened latent z need to reshape to [batch, H, W, C]
        dense_z = self.relu_d(self.dense_1(z))
        z_3d = tf.reshape(dense_z, [-1, 15, 15, 1])
        h1 = self.relu_1(self.bn_1(self.deconv_1(z_3d), training=training))
        h2 = self.relu_2(self.bn_2(self.deconv_2(h1), training=training))
        h3 = self.relu_3(self.bn_3(self.deconv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.deconv_4(h3), training=training))
        h5 = tf.keras.activations.sigmoid(self.bn_5(self.deconv_5(h4), training=training))  # img re-scaled to vals [0-1] -> sigmoid output used for decoder last layer
        return h5


class ConvDecoder(tf.keras.Model):

    def __init__(self, colour, img_size):
        super(ConvDecoder, self).__init__()
        self.img_size = img_size
        self.dense_1 = tf.keras.layers.Dense(int(self.img_size/4)*int(self.img_size/4))
        self.relu_d = tf.keras.layers.ReLU()

        self.conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                             padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                             padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                             padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.conv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                             padding="same")
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        self.conv_5 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=1,
                                             padding="same")
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.relu_5 = tf.keras.layers.ReLU()

        if colour:  # color image
            self.conv_6 = tf.keras.layers.Conv2D(filters=3, kernel_size=4, strides=1,
                                                 padding="same")
            self.bn_6 = tf.keras.layers.BatchNormalization()
        elif not colour:  # greyscale image
            self.conv_6 = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1,
                                                 padding="same")
            self.bn_6 = tf.keras.layers.BatchNormalization()

    def call(self, z, training=True):
        # flattened latent z need to reshape to [batch, H, W, C]
        dense_z = self.relu_d(self.dense_1(z))
        z_3d = tf.reshape(dense_z, [-1, int(self.img_size/4), int(self.img_size/4), 1])
        h1 = self.relu_1(self.bn_1(self.conv_1(z_3d), training=training))  # (21, 21)
        # upsample 2x
        h1_shape = h1.shape
        h1 = tf.image.resize(h1, (2*h1_shape[1], 2*h1_shape[2]), method='nearest')
        h2 = self.relu_2(self.bn_2(self.conv_2(h1), training=training))  # (42, 42)
        # upsample 2x
        h2_shape = h2.shape
        h2 = tf.image.resize(h2, (2*h2_shape[1], 2*h2_shape[2]), method='nearest')  # (84, 84)
        h3 = self.relu_3(self.bn_3(self.conv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.conv_4(h3), training=training))
        h5 = self.relu_5(self.bn_5(self.conv_5(h4), training=training))
        h6 = tf.keras.activations.sigmoid(self.bn_6(self.conv_6(h5), training=training))  # img re-scaled to vals [0-1] -> sigmoid output used for decoder last layer
        return h6


class Pnet(tf.keras.Model):

    def __init__(self, num_keypoints, gauss_std):
        super(Pnet, self).__init__()
        self.num_keypoints = num_keypoints
        self.gauss_std = gauss_std

        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=4,
                                             strides=1, padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                             strides=2, padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                             strides=2, padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.layers.ReLU()

        self.conv_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3,
                                             strides=1, padding="same")
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.layers.ReLU()

        self.conv_5 = tf.keras.layers.Conv2D(filters=self.num_keypoints,
                                           kernel_size=1, strides=1, padding="same")

    def call(self, inputs, training=True):
        h1 = self.relu_1(self.bn_1(self.conv_1(inputs), training=training))
        h2 = self.relu_2(self.bn_2(self.conv_2(h1), training=training))
        h3 = self.relu_3(self.bn_3(self.conv_3(h2), training=training))
        h4 = self.relu_4(self.bn_4(self.conv_4(h3), training=training))
        keypoint_features = self.conv_5(h4)
        return get_keypoint_data_from_feature_map(keypoint_features, self.gauss_std)
