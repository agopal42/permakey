import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import preprocess
from vision_module import TransporterEncoder
from vision_module import TransporterKeypointer
from vision_module import TransporterDecoder
import ul_loss

from utils import add_sacred_log
from sacred import Experiment
import json

ex = Experiment()


@ex.config
def config():
    # Training
    learning_rate = 0.0002
    decay_rate = 0.9
    decay_steps = 30000
    batch_size = 64
    epochs = 100
    env = "mspacman"
    loss_to_use = "transporter"
    num_keypoints = 7
    gauss_std = 0.1  # std-dev of gaussian window around keypoint location
    max_patience = 10  # early-stopping param

    # Architecture
    img_input = "dm_atari"
    noise_type = "none"  # "vertical", "horizontal", "both", "none"
    colour_input = True

    # Logging
    data_dir = "../mutual_information_keypoints/mini_data/"
    gpu = 0

    # Eval
    eval_split = "test"


@ex.capture
def create_model(colour_input, num_keypoints, gauss_std, loss_to_use):
    encoder, keypointer, decoder = None, None, None
    if loss_to_use == "transporter":
        encoder = TransporterEncoder()
        keypointer = TransporterKeypointer(num_keypoints=num_keypoints, gauss_std=gauss_std)
        decoder = TransporterDecoder(colour=colour_input)

    else:
        print("loss type %s not supported" % loss_to_use)
    return encoder, keypointer, decoder


@ex.command
def evaluate(data_dir, env, ckpt_load_dir, test_logs_prefix, loss_to_use, noise_type, eval_split,
            img_input, colour_input, num_keypoints, gauss_std, batch_size, epoch, _run):

    test_inputs, keypoints, heatmaps, x_pred = 0.0, 0.0, 0.0, 0.0
    encoder, keypointer, decoder = None, None, None

    # setup data pipeline
    if img_input == "dm_atari":
        eval_dataset = preprocess.deepmind_atari(
            data_dir, env, eval_split, loss_to_use, batch_size, noise_type,
            colour_input)
    else:
        raise ValueError("Eval data %s does not exist" % img_input)

    # load best ckpt models
    if loss_to_use == "transporter":
        encoder = TransporterEncoder()
        keypointer = TransporterKeypointer(num_keypoints=num_keypoints, gauss_std=gauss_std)
        decoder = TransporterDecoder(colour_input)

    # FIX: run 1 forward pass over models to make it do weight init
    if colour_input:
        test_inputs = tf.zeros((batch_size, 84, 84, 3, 2))
    if not colour_input:
        test_inputs = tf.zeros((batch_size, 84, 84, 1, 2))
    _ = ul_loss.transporter_loss(test_inputs, encoder, keypointer, decoder, training=True)

    # restore best model weights
    encoder.load_weights(ckpt_load_dir + 'encoder-' + str(epoch) + '.h5')
    decoder.load_weights(ckpt_load_dir + 'decoder-' + str(epoch) + '.h5')
    keypointer.load_weights(ckpt_load_dir + 'keypointer-' + str(epoch) + '.h5')

    batch_num = 0
    test_recon_loss = 0.0
    for x_test in eval_dataset:
        batch_num = batch_num + 1
        if loss_to_use == "transporter":
            keypoints, heatmaps, features, x_pred, loss = ul_loss.transporter_loss(x_test,
                                    encoder, keypointer, decoder, training=False)

            test_recon_loss = test_recon_loss + loss

        # saving data
        if not (os.path.exists(test_logs_prefix)):
            # create the directory you want to save to
            os.makedirs(test_logs_prefix)
        np.savez(test_logs_prefix + "/" + "epoch_" + str(epoch) + "_batch_" +
                 str(batch_num) + "_keypoints.npz", x_pred, x_test.numpy(), keypoints, heatmaps)

    # log test loss
    test_recon_loss = test_recon_loss / batch_num

    # logging avg. test epoch losses to Sacred
    add_sacred_log("test.epoch_recon_loss", float(test_recon_loss.numpy()), _run)

    print(" avg. test_nll_loss: %3.4f " % (test_recon_loss.numpy()))
    return 0.


@ex.capture
def train(img_input, data_dir, env, batch_size, loss_to_use, decay_steps, decay_rate,
          max_patience, colour_input, noise_type, learning_rate, epochs,
          checkpoint_prefix, _run):

    # setup data pipeline
    if img_input == "dm_atari":
        train_dataset = preprocess.deepmind_atari(
            data_dir, env, "train", loss_to_use, batch_size, noise_type,
            colour_input)
        valid_dataset = preprocess.deepmind_atari(
            data_dir, env, "valid", loss_to_use, batch_size, noise_type,
            colour_input)
    else:
        raise ValueError("Input data %s does not exist" % img_input)

    # create models
    if loss_to_use == "transporter":
        encoder, keypointer, decoder = create_model()

    # setting up checkpointing and summaries
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate,
                                           decay_steps, decay_rate, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)

    def train_step(images, loss_type):
        if loss_type == "transporter":
            with tf.GradientTape() as tape:
                reconstruction_loss = ul_loss.transporter_loss(images,
                                encoder, keypointer, decoder, training=True)

            # update params
            model_params = encoder.trainable_variables + keypointer.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(reconstruction_loss, model_params)
            optimizer.apply_gradients(zip(grads, model_params))

        return reconstruction_loss

    def test_step(images, loss_type):
        if loss_type == "transporter":
            keypoints, heatmaps, features, x_hat, loss = ul_loss.transporter_loss(images,
                                encoder, keypointer, decoder, training=False)
            return keypoints, heatmaps, x_hat, loss

    # training
    best_validation_loss, best_validation_epoch = float("inf"), -1  # val_loss, val_epoch
    patience = 0
    step = 0
    for epoch in range(epochs):
        total_recon_loss = 0.0
        num_batches = 0
        # TRAIN LOOP
        start_time_epoch = time.time()
        i = 0
        for x_train in train_dataset:
            start_time = time.time()
            loss = train_step(x_train, loss_to_use)

            print("batch number: %4d reconstruction_loss: %4.5f took %4.5f s" %
                  (num_batches, loss.numpy(), time.time() - start_time))

            total_recon_loss = total_recon_loss + loss
            num_batches += 1
            step += 1

            # logging train vae and pm losses to Sacred
            add_sacred_log("train.step_reconstruction_loss", float(loss.numpy()), _run)
            add_sacred_log("train.step", step, _run)

        epoch_recon_loss = total_recon_loss / num_batches
        batch_per_second = num_batches / (time.time() - start_time_epoch)

        # logging avg. epoch losses to Sacred
        add_sacred_log("train.epoch_reconstruction_loss", float(epoch_recon_loss.numpy()), _run)
        add_sacred_log("train.epoch", epoch, _run)

        # VALIDATION LOOP
        # end of every epoch compute validation loss and checkpoint models based on that
        total_valid_recon_loss = 0.0
        valid_num_batch = 0
        for x_valid in valid_dataset:
            keypoints, heatmaps, x_hat, valid_batch_loss = test_step(x_valid, loss_to_use)

            total_valid_recon_loss = total_valid_recon_loss + valid_batch_loss
            valid_num_batch = valid_num_batch + 1

        epoch_val_recon_loss = total_valid_recon_loss / valid_num_batch
        # printing out avg. train  end of every epoch
        print("end of epoch: %2d avg. train_recon_loss: %3.4f avg. batch/s: %3.4f" % (
            epoch, epoch_recon_loss.numpy(), batch_per_second))
        # printing out avg.validation losses
        print("end of epoch: %2d avg. val_recon_loss: %3.4f batch/s: %3.4f" % (
            epoch, epoch_val_recon_loss.numpy(), batch_per_second))

        # logging validation_losses to Sacred
        add_sacred_log("validation.epoch_nll_loss", float(epoch_val_recon_loss.numpy()), _run)
        add_sacred_log("validation.epoch", epoch, _run)

        # checkpointing models based on validation loss
        validation_loss = epoch_val_recon_loss  # epoch_val_nll_loss + epoch_val_kl_loss +
        if validation_loss.numpy() < best_validation_loss:
            # update best_validation loss
            best_validation_loss, best_validation_epoch = validation_loss.numpy(), epoch
            encoder.save_weights(checkpoint_prefix + '_encoder-' + str(best_validation_epoch) + '.h5')
            decoder.save_weights(checkpoint_prefix + '_decoder-' + str(best_validation_epoch) + '.h5')
            keypointer.save_weights(checkpoint_prefix + '_keypointer-' + str(best_validation_epoch) + '.h5')
            # early_stopping param resets
            patience = 0

        # early stopping check
        elif validation_loss.numpy() >= best_validation_loss:
            patience = patience + 1

        # break out if max_patience is reached
        if patience == max_patience:
            break

    print("Training complete!! Best validation loss : %3.4f achieved at epoch: %2d"
          % (best_validation_loss, best_validation_epoch))

    add_sacred_log("validation.best_val_loss", float(best_validation_loss), _run)
    add_sacred_log("validation.best_val_epoch", best_validation_epoch, _run)
    return best_validation_loss, best_validation_epoch


@ex.automain
def main(gpu, img_input, noise_type, env, num_keypoints, _config):

    if not 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # init folder for logging images/checkpoints
    train_dir = os.path.join("transporter_exp", img_input, noise_type, env,
                             str(num_keypoints), datetime.now().isoformat())
    ckpt_prefix = os.path.join(train_dir, "ckpt")
    test_logs_prefix = os.path.join(train_dir, "test_logs")
    # create dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # dump experiment config to json file
    flags_json_fn = os.path.join(train_dir, 'flags.json')
    with open(flags_json_fn, 'w') as outfile:
        json.dump(_config, outfile, indent=4)
        print('Wrote config to json file: ', flags_json_fn)

    # run train
    best_validation_loss, best_validation_epoch = train(checkpoint_prefix=ckpt_prefix)
    # ckpt load directory
    ckpt_load_dir = train_dir + "/ckpt_"  # ckpt indexed from 1 but epochs from 0
    # run evaluate
    evaluate(ckpt_load_dir=ckpt_load_dir, epoch=best_validation_epoch,
            test_logs_prefix=test_logs_prefix)
    return best_validation_loss
