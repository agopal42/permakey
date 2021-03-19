import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class LspMlp(tf.keras.Model):

    def __init__(self, input_size, output_size):
        super(LspMlp, self).__init__()
        self.W_b_0 = tf.keras.layers.Dense(units=input_size)
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.relu_0 = tf.keras.layers.ReLU()

        self.W_b_1 = tf.keras.layers.Dense(units=512)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.W_b_2 = tf.keras.layers.Dense(units=256)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()

        self.W_b_3 = tf.keras.layers.Dense(units=output_size, activation="linear")

    def call(self, inputs, training=True):
        h0 = self.relu_0(self.bn_0(self.W_b_0(inputs), training=training))
        h1 = self.relu_1(self.bn_1(self.W_b_1(h0), training=training))
        h2 = self.relu_2(self.bn_2(self.W_b_2(h1), training=training))
        h3 = self.W_b_3(h2)
        return h3


def normalize_img(inputs):
    # perform per-image 0-1 normalization
    min_val = tf.expand_dims(tf.expand_dims(tf.math.reduce_min(inputs, axis=[1, 2]), axis=1), axis=2)
    max_val = tf.expand_dims(tf.expand_dims(tf.math.reduce_max(inputs, axis=[1, 2]), axis=1), axis=2)
    norm_inputs = tf.math.divide((inputs - min_val), (max_val - min_val))
    return norm_inputs


def extract_neighbouring_patches(activation, patch_cell_size):
    """
    Extracts 3 x 3 patches from an image, such that every non-overlapping
    patch_cell_size x patch_cell_size region becomes a center.
    :param activation: Input map (B, H, W, D)
    :param patch_cell_size: Size of each patch cell
    :return: center patch cells and outer patch cells
    """
    patch_size = (3 * patch_cell_size, 3 * patch_cell_size)

    # Pad to ensure each grid cell is a center.
    padded_activations = tf.pad(activation, [
        [0, 0],
        [patch_cell_size, patch_cell_size],
        [patch_cell_size, patch_cell_size],
        [0, 0]])

    patches = tf.image.extract_patches(
        padded_activations, sizes=[1, patch_size[0], patch_size[1], 1],
                            strides=[1, patch_cell_size, patch_cell_size, 1],
                            rates=[1, 1, 1, 1], padding="VALID")

    # Unpack to (B, H, W, 9 * P * P, D)
    patches = tf.reshape(patches, [
        patches.shape[0], patches.shape[1], patches.shape[2],
        patch_size[0] * patch_size[1], activation.shape[-1]])

    patches = tf.split(patches, 9 * patch_cell_size, axis=3)

    # Gather center patches
    center_patch_cell = tf.concat(
        patches[3 * patch_cell_size + 1: 6 * patch_cell_size + 1:3], axis=3)

    # Gather outer patches
    outer_patch_cells = patches[:3 * patch_cell_size]
    for i in range(patch_cell_size):
        outer_patch_cells.append(patches[3 * (patch_cell_size + i)])
        outer_patch_cells.append(patches[3 * (patch_cell_size + i) + 2])
    outer_patch_cells.extend(patches[6 * patch_cell_size:])
    outer_patch_cells = tf.concat(outer_patch_cells, axis=3)

    return center_patch_cell, outer_patch_cells


def lsp_loss(lsp_nets, enc_activations, patch_sizes, img_size, lsp_layers,
             pnet=None, training=True):
    """
    :param lsp_nets: list of lsp networks (1 for each layer of cnn) used to
    compute patch-level lsp
    :param enc_activations: list of layer-wise conv_encoder activations
    :param training:
    :return:
    :loc_sp_loss: local spatial prediction loss computed on localized patches of activation maps
    :mask : error masks (saliency masks) if Training=False
    """
    loc_sp_loss = tf.zeros((1), dtype=tf.float32)
    error_mask = []
    resized_error_mask = []
    # compute pm_loss at specified encoder layers
    for l in range(len(lsp_layers)):
        # computing local spatial prediction at chosen encoder layers
        patch_size = patch_sizes[l]
        lsp_layer = lsp_layers[l]
        batch_size = enc_activations[0].shape[0]

        # aggregating the patches of 1st-order neighbours
        centers, neighbours = extract_neighbouring_patches(
            enc_activations[lsp_layer], patch_size)

        # reshape (B, H, W, P * P, D) -> (B,  H * W,  P * P *  D)
        n_patches = centers.shape[1]*centers.shape[2]
        centers = tf.reshape(centers, [
            batch_size, centers.shape[1], centers.shape[2], -1])
        centers = tf.reshape(centers, [batch_size, n_patches, centers.shape[3]])

        # reshape (B, H, W, 8 * P * P, D) -> (B * H * W, 8 * P * P * D)
        neighbours = tf.reshape(neighbours, [
            batch_size, neighbours.shape[1], neighbours.shape[2], -1])
        neighbours = tf.reshape(neighbours, [-1, neighbours.shape[3]])

        # Make predictions
        center_preds = lsp_nets[l](neighbours, training=training)
        center_preds = tf.reshape(center_preds, [batch_size, n_patches, -1])
        centers = tf.reshape(centers, [batch_size, n_patches, -1])
        # lsp-loss
        errors = tf.math.square(centers - center_preds)
        loc_sp_loss = loc_sp_loss + tf.reduce_mean(errors)

        errors_2d = tf.math.sqrt(tf.reduce_mean(errors, axis=2))  # RMSE
        # normalize error maps w.r.t ground-truth values
        patch_gt_2d = tf.reduce_mean(centers, axis=2)
        eps = tf.constant(10**-3, dtype=tf.float32)
        norm_errors_2d = tf.math.divide(errors_2d, patch_gt_2d + eps)
        map_shape = enc_activations[lsp_layer].shape
        error_mask.append(tf.reshape(norm_errors_2d, [batch_size,
                        tf.math.floordiv(map_shape[1], patch_size),
                        tf.math.floordiv(map_shape[2], patch_size)]))
        # resize all error masks to same size
        resized_error_mask.append(tf.image.resize(tf.expand_dims(error_mask[l], axis=3),
                                                  (img_size, img_size), method='nearest'))

    # compute pnet_loss
    pnet_loss = tf.zeros((1), dtype=tf.float32)
    # stack error maps in channel dim
    resized_error_mask_tensor = tf.convert_to_tensor(resized_error_mask, dtype=tf.float32)
    stacked_e_mask = [resized_error_mask_tensor[i, :, :, :, :] for i in range(len(lsp_layers))]
    stacked_error_mask = tf.concat(stacked_e_mask, axis=3)
    # 0-1 normalize error maps
    stacked_error_mask = normalize_img(stacked_error_mask)
    # use pnet to get keypoints from different hierarchies of error maps
    keypoints, heatmaps = pnet(stacked_error_mask, training=training)

    if training:
        # sum heatmaps in channel dim
        heatmaps_sum = tf.image.resize(tf.expand_dims(tf.reduce_sum(heatmaps, axis=3), axis=3),
                                       (img_size, img_size), method='nearest')
        # compute pnet loss
        pnet_loss = tf.reduce_mean(tf.math.square(heatmaps_sum - stacked_error_mask))

    if training:
        return loc_sp_loss, pnet_loss
    elif not training:
        return keypoints, heatmaps, stacked_error_mask, loc_sp_loss, pnet_loss


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = tf.math.square(tf.math.subtract(preds, target))
    # normalizing by whole img.
    avg_neg_log_p = tf.math.divide(tf.reduce_mean(neg_log_p, axis=[1, 2, 3]), 2*variance)
    if add_const:
        const = 0.5*tf.math.log(2. * np.pi * variance)
        avg_neg_log_p += const
    return avg_neg_log_p


def pkey_loss(model_list, x, latent_dim_size, patch_sizes, batch_size,
                       img_size, lsp_layers, loss_type, training):
    """
    :param model_list: list of neural network modules used in PermaKey system
    :param x: batch of input images
    :param latent_dim_size: (int) size of VAE latent dim
    :param patch_sizes: (int) patch size used for LSP
    :param batch_size: (int) minibatch size
    :param img_size: (int) size of input image (assumed to be square H x H)
    :param lsp_layers: (tuple) choice of CNN feature layers for LSP
    :param loss_type: (str)
    :param training: (bool) True during training phase (False) at inference time
    :return:
    """
    # unpack models from model_list
    encoder, decoder, lsp_nets, pnet = model_list[0], model_list[1], model_list[2], model_list[3]

    # encoder forward pass to get q(z | x)
    mu, var, enc_activations = encoder(x, training=training)
    # posterior distribution
    q_z = tfp.distributions.Normal(loc=mu, scale=var)
    assert q_z.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED
    # prior
    p_z = tfp.distributions.Normal(loc=tf.zeros(latent_dim_size, dtype=tf.float32)
                                   , scale=tf.ones(latent_dim_size, dtype=tf.float32))
    # kl loss
    kl = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    # reconstruction decoder forward pass i.e. p_psi(x_pred | z)
    z = q_z.sample()
    x_pred = decoder(z, training=training)
    # compute reconstruction loss
    variance = 10**-1*tf.ones(batch_size,)
    l2_loss = nll_gaussian(x_pred, x, variance, add_const=False)

    # averaging over batch
    nll_loss = tf.reduce_mean(l2_loss, 0)
    kl_loss = tf.reduce_mean(kl, 0)

    if training:
        # compute loc_sp_loss (LSP task) => predict patch of cnn activations given activations of neighbours
        loc_sp_loss, pnet_loss = lsp_loss(lsp_nets, enc_activations, patch_sizes,
                                 img_size, lsp_layers, pnet, training=training)
        # returns only losses in training mode
        return [nll_loss, kl_loss, loc_sp_loss, pnet_loss]

    # inference time
    elif not training:
        kpts, gauss_mask, error_mask, loc_sp_loss, pnet_loss = lsp_loss(lsp_nets,
                                    enc_activations, patch_sizes, img_size,
                                    lsp_layers, pnet, training=training)
        # returns losses + error_masks + reconstructions in inference mode
        loss_list = [nll_loss, kl_loss, loc_sp_loss, pnet_loss]
        return x_pred, kpts, gauss_mask, error_mask, loss_list


def transporter_loss(images, encoder, keypointer, decoder, training=True):
    """
    :params:
    images = Tensor of shape [B, H, W, C, 2] at training containing a batch of images.
             Tensor of shape [B, H, W, C] at test time containing a batch of images.
    encoder: conv encoder network tf.keras.Model class object
    keypointer: keypoint network (PointNet) tf.keras.Model class object
    decoder: conv decoder network tf.keras.Model class object
    training: `bool` indication whether the model is in training mode.

    :return:
    loss = reconstruction loss from Transporter paper https://arxiv.org/abs/1906.11883
    """

    image_a = images[:, :, :, :, 0]
    image_b = images[:, :, :, :, 1]

    # Process both images. All gradients related to image_a are stopped.
    image_a_features = tf.stop_gradient(encoder(image_a, training=training))
    image_a_keypoints, image_a_heatmaps = keypointer(image_a, training=training)

    # stop_gradient fix for list of tensors of different shapes
    image_a_keypoints = tf.stop_gradient(image_a_keypoints)
    image_a_heatmaps = tf.stop_gradient(image_a_heatmaps)

    image_b_features = encoder(image_b, training=training)
    image_b_keypoints, image_b_heatmaps = keypointer(image_b, training=training)

    # Transport features
    num_keypoints = image_a_heatmaps.shape[-1]
    transported_features = image_a_features
    for k in range(num_keypoints):
        mask_a = image_a_heatmaps[Ellipsis, k, None]
        mask_b = image_b_heatmaps[Ellipsis, k, None]

        # suppress features from image a, around both keypoint locations.
        transported_features = ((1 - mask_a)*(1 - mask_b)*transported_features)

        # copy features from image b around keypoints for image b.
        transported_features += (mask_b*image_b_features)

    reconstructed_image_b = decoder(transported_features, training=training)

    # avg. over minibatch
    l2_loss = tf.reduce_mean(tf.math.square(tf.math.subtract(image_b, reconstructed_image_b)))
    if training:
        return l2_loss

    if not training:
        return image_b_keypoints, image_b_heatmaps, image_b_features, \
               reconstructed_image_b, l2_loss
