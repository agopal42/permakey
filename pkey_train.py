import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import preprocess
from vision_module import ConvEncoder, DeconvDecoder, ConvDecoder, Pnet
import ul_loss
from ul_loss import LspMlp
import transporter_train
from utils import add_sacred_log
from sacred import Experiment
import json


ex = Experiment()


@ex.config
def config():
    # Training
    learning_rate = 0.0002
    decay_rate = 0.85
    decay_steps = 10000
    batch_size = 32
    epochs = 100
    env = "mspacman"
    loss_to_use = "pkey"
    lsp_layers = (0, 1)
    latent_dim_size = 32
    patch_sizes = (2, 2)
    num_keypoints = 7
    gauss_std = 0.12
    max_patience = 10  # early-stopping param

    # Architecture
    decoder_type = "conv"  # ("deconv", "conv")
    img_input = "dm_atari"  # ("dm_atari" for atari envs)
    img_size = 84  # atari: 84
    noise_type = "none"  # "vertical", "horizontal", "both", "none"
    colour_input = True

    # Logging
    data_dir = "../mutual_information_keypoints/mini_data/"  # atari: data/atari/
    test_every = 1
    save_every = 1
    load = 0
    gpu = 0

    # Eval
    eval_split = "test"
    tp_epoch = -1  # ckpt-epoch of transporter model to evaluate qualitatively
    pkey_epoch = -1  # ckpt-epoch of permakey model to evaluate qualitatively
    save_base_dir = "compare_kpts/"
    ablation = False  # indicates eval for pkey ablations (True) or model comparison (False)


@ex.capture
def create_model(decoder_type, latent_dim_size, colour_input,
                 loss_to_use, patch_sizes, lsp_layers, num_keypoints,
                 gauss_std, img_size):

    encoder, decoder, lsp_models, pnet, model_list = None, None, [], None, []

    encoder = ConvEncoder(latent_dim_size)

    if decoder_type == "deconv":
        decoder = DeconvDecoder(colour_input)
    elif decoder_type == "conv":
        decoder = ConvDecoder(colour_input, img_size)
    else:
        print("decoder type %s not supported" % decoder_type)

    if loss_to_use == "pkey":
        for l in range(len(lsp_layers)):
            output_size = patch_sizes[l]**2*encoder.filters[lsp_layers[l]]
            lsp_model = LspMlp(8*output_size, output_size)
            lsp_models.append(lsp_model)

        pnet = Pnet(num_keypoints, gauss_std)
        model_list = [encoder, decoder, lsp_models, pnet]
    else:
        print("loss type %s not supported" % loss_to_use)
    return model_list


@ex.command
def compare_kpts(data_dir, loss_to_use, num_keypoints, latent_dim_size, env,
                 img_input, img_size, colour_input, patch_sizes, lsp_layers,
                 noise_type, batch_size, eval_split, tp_fname, pkey_fname,
                 tp_epoch, pkey_epoch, save_base_dir, ablation, _run):
 
    # Input params
    tp_ckpt_load_dir = "transporter_exp/" + img_input + "/" + noise_type \
                       + "/" + env + "/" + str(num_keypoints) + "/" + \
                       tp_fname + "/ckpt_"
    pkey_ckpt_load_dir = "permakey_exp/" + img_input + "/" + noise_type \
                         + "/" + env + "/" + str(num_keypoints) + "/" + \
                         pkey_fname + "/ckpt_"

    model_id = ""
    if not ablation:
        # numerical string after '.' as unique model_id
        model_id = pkey_ckpt_load_dir.split(".")[1][0:6] + "_" \
                   + tp_ckpt_load_dir.split(".")[1][0:6]
    elif ablation:
        model_id = pkey_ckpt_load_dir.split(".")[1][0:6]

    save_dir = save_base_dir + img_input + "/" + noise_type + "/" \
               + env + "/" + str(num_keypoints) + "/" + model_id

    # setup data pipeline
    if img_input == "dm_atari":
        eval_dataset = preprocess.deepmind_atari(data_dir, env, eval_split,
                                                 loss_to_use, batch_size,
                                                 noise_type, colour_input)
    else:
        raise ValueError("Eval data %s does not exist" % img_input)

    # load best pkey ckpt models
    pkey_model_list = create_model()
    tp_kp_model_list = transporter_train.create_model(colour_input,
                                            num_keypoints, 0.1, "transporter")

    # unpacking models from model list
    encoder, decoder, lsp_models, pnet = pkey_model_list[0], pkey_model_list[1], \
                                        pkey_model_list[2], pkey_model_list[3]
    # FIX: run 1 forward pass over models to make it do weight init
    if colour_input:
        test_inputs = tf.zeros((batch_size, img_size, img_size, 3))
    if not colour_input:
        test_inputs = tf.zeros((batch_size, img_size, img_size, 1))
    _ = ul_loss.pkey_loss(pkey_model_list, test_inputs, latent_dim_size,
                                  patch_sizes, batch_size, img_size, lsp_layers,
                                  loss_to_use, training=True)
 
    # restore best model weights
    encoder.load_weights(pkey_ckpt_load_dir + 'encoder-' + str(pkey_epoch) + '.h5')
    decoder.load_weights(pkey_ckpt_load_dir + 'decoder-' + str(pkey_epoch) + '.h5')
    pnet.load_weights(pkey_ckpt_load_dir + 'pnet-' + str(pkey_epoch) + '.h5')
    for m in range(len(lsp_models)):
        lsp_models[m].load_weights(pkey_ckpt_load_dir + 'lsp_model-layer-' +
                                   str(lsp_layers[m]) + '-' + str(pkey_epoch) + '.h5')

    pkey_model_list = [encoder, decoder, lsp_models, pnet]

    # unpacking models from tp_model_list
    tp_encoder, keypointer, decoder = tp_kp_model_list[0], tp_kp_model_list[1], tp_kp_model_list[2]

    if colour_input:
        test_inputs = tf.zeros((batch_size, img_size, img_size, 3, 2))
    if not colour_input:
        test_inputs = tf.zeros((batch_size, img_size, img_size, 1, 2))
    _ = ul_loss.transporter_loss(test_inputs, tp_encoder, keypointer, decoder, training=True)

    # restore best model weights
    tp_encoder.load_weights(tp_ckpt_load_dir + 'encoder-' + str(tp_epoch) + '.h5')
    decoder.load_weights(tp_ckpt_load_dir + 'decoder-' + str(tp_epoch) + '.h5')
    keypointer.load_weights(tp_ckpt_load_dir + 'keypointer-' + str(tp_epoch) + '.h5')

    batch_num = 0
    for x_test in eval_dataset:
        batch_num = batch_num+1
        # inference using pkey model
        x_pred, kpts, gauss_mask, error_mask, _ = ul_loss.pkey_loss(pkey_model_list,
                                                x_test, latent_dim_size, patch_sizes,
                                                batch_size, img_size, lsp_layers,
                                                loss_to_use, training=False)

        # inference using tp_model
        tp_x_test = tf.stack([x_test, x_test], axis=4)
        kpts_tp, gauss_mask_tp, features, x_pred_tp, _ = ul_loss.transporter_loss(tp_x_test,
                                                tp_encoder, keypointer, decoder, training=False)

        # logging results for viz
        if not (os.path.exists(save_dir)):
            # create the directory you want to save to
            os.makedirs(save_dir)
        # saving data from pkey model
        np.savez(save_dir + "/" + "batch_" + str(batch_num) + "_preds_masks.npz", x_pred, x_test.numpy(), kpts, gauss_mask, error_mask)
        # saving data from tp_model
        np.savez(save_dir + "/" + "batch_" + str(batch_num) + "_keypoints.npz", x_pred_tp, x_test.numpy(), kpts_tp, gauss_mask_tp)
    return 0


@ex.command
def evaluate(ckpt_load_dir, test_logs_prefix, data_dir, loss_to_use, img_size,
             latent_dim_size, env, img_input, colour_input, patch_sizes,
             lsp_layers, noise_type, batch_size, epoch, eval_split, _run):

    # setup data pipeline
    if img_input == "dm_atari":
        eval_dataset = preprocess.deepmind_atari(data_dir, env, eval_split, loss_to_use,
                                            batch_size, noise_type, colour_input)
    else:
        raise ValueError("Eval data %s does not exist" % img_input)

    # load best ckpt models
    model_list = create_model()
    # unpacking models from model list
    encoder, decoder, lsp_models, pnet = model_list[0], model_list[1], model_list[2],\
                                         model_list[3]
    # FIX: run 1 forward pass over models to make it do weight init
    if colour_input:
         test_inputs = tf.zeros((batch_size, img_size, img_size, 3))
    if not colour_input:
         test_inputs = tf.zeros((batch_size, img_size, img_size, 1))
    _ = ul_loss.pkey_loss(model_list, test_inputs, latent_dim_size, patch_sizes,
                            batch_size, img_size, lsp_layers, loss_to_use, training=True)

    # restore best model weights
    encoder.load_weights(ckpt_load_dir + 'encoder-' + str(epoch) + '.h5')
    decoder.load_weights(ckpt_load_dir + 'decoder-' + str(epoch) + '.h5')
    pnet.load_weights(ckpt_load_dir + 'pnet-' + str(epoch) + '.h5')
    for m in range(len(lsp_models)):
        lsp_models[m].load_weights(ckpt_load_dir + 'lsp_model-layer-' + str(lsp_layers[m]) + '-' + str(epoch) + '.h5')

    model_list = [encoder, decoder, lsp_models, pnet]

    batch_num = 0
    test_nll_loss = 0.0
    test_kl_loss = 0.0
    test_lsp_loss = 0.0
    test_pnet_loss = 0.0

    for x_test in eval_dataset:
        batch_num = batch_num + 1
        x_pred, kpts, gauss_mask, error_mask, loss = ul_loss.pkey_loss(model_list, x_test,
                                                       latent_dim_size, patch_sizes,
                                                       batch_size, img_size,
                                                       lsp_layers, loss_to_use,
                                                       training=False)
        nll_loss, kl_loss, lsp_loss, pnet_loss = loss[0], loss[1], loss[2], loss[3]

        test_nll_loss = test_nll_loss + nll_loss
        test_kl_loss = test_kl_loss + kl_loss
        test_lsp_loss = test_lsp_loss + lsp_loss
        test_pnet_loss = test_pnet_loss + pnet_loss
        # saving data
        if not (os.path.exists(test_logs_prefix)):
            # create the directory you want to save to
            os.makedirs(test_logs_prefix)
        np.savez(test_logs_prefix + "/" + "epoch_" + str(epoch) + "_batch_"
        + str(batch_num) + "_preds_masks.npz", x_pred, x_test.numpy(), kpts, gauss_mask, error_mask)

    # log test loss
    test_nll_loss = test_nll_loss / batch_num
    test_kl_loss = test_kl_loss / batch_num
    test_lsp_loss = test_lsp_loss / batch_num
    test_pnet_loss = test_pnet_loss / batch_num

    # logging avg. test epoch losses to Sacred
    add_sacred_log("test.epoch_nll_loss", float(test_nll_loss.numpy()), _run)
    add_sacred_log("test.epoch_kl_loss", float(test_kl_loss.numpy()), _run)
    add_sacred_log("test.epoch_lsp_loss", float(test_lsp_loss.numpy()), _run)
    add_sacred_log("test.epoch_pnet_loss", float(test_pnet_loss.numpy()), _run)

    print("%s: avg._nll_loss: %3.4f avg. kl_loss: %3.4f avg. lsp_loss: %3.4f avg. pnet_loss: %3.4f"
        % (eval_split, test_nll_loss.numpy(), test_kl_loss.numpy(), test_lsp_loss.numpy(), test_pnet_loss.numpy()))
    # checkpointing models based on validation loss
    loss = test_pnet_loss + test_lsp_loss
    return loss


@ex.capture
def train(img_input, data_dir, env, batch_size, loss_to_use, img_size, lsp_layers,
        latent_dim_size, max_patience, colour_input, noise_type, patch_sizes,
        learning_rate, decay_rate, decay_steps, epochs, checkpoint_prefix, _run):

    # setup data pipeline
    if img_input == "dm_atari":
        train_dataset = preprocess.deepmind_atari(data_dir, env, "train", loss_to_use, batch_size, noise_type, colour_input)
        valid_dataset = preprocess.deepmind_atari(data_dir, env, "valid", loss_to_use, batch_size, noise_type, colour_input)
    else:
        raise ValueError("Input data %s does not exist" % img_input)

    # create models
    model_list = create_model()
    # unpacking models
    encoder, decoder, lsp_models, pnet = model_list[0], model_list[1], model_list[2], model_list[3]

    # setting up optimizer and decay params
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(learning_rate, decay_steps, decay_rate, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)

    def train_step(inputs, loss_type):
        with tf.GradientTape() as vae_tape, tf.GradientTape(persistent=True) as lsp_tape, tf.GradientTape() as pnet_tape:
            loss_list = ul_loss.pkey_loss(model_list, inputs, latent_dim_size,
                        patch_sizes, batch_size, img_size, lsp_layers, loss_type, training=True)

            nll_loss, kl_loss, lsp_loss, pnet_loss = loss_list[0], loss_list[1], loss_list[2], loss_list[3]
            vae_loss = nll_loss + kl_loss

        # vae update
        # opt_start = time.time()
        vae_params = encoder.trainable_variables + decoder.trainable_variables
        vae_grads = vae_tape.gradient(vae_loss, vae_params)
        optimizer.apply_gradients(zip(vae_grads, vae_params))
        # lsp update
        for l in range(len(lsp_models)):
            lsp_params = lsp_models[l].trainable_variables
            lsp_grads = lsp_tape.gradient(lsp_loss, lsp_params)
            optimizer.apply_gradients(zip(lsp_grads, lsp_params))

        pnet_params = pnet.trainable_variables
        pnet_grads = pnet_tape.gradient(pnet_loss, pnet_params)
        optimizer.apply_gradients(zip(pnet_grads, pnet_params))
        # print("weight updates took %4.5f" % (time.time() - opt_start))
        return loss_list

    def test_step(inputs):
        x_hat, kpts, gauss_mask, error_mask, loss = ul_loss.pkey_loss(model_list, inputs,
                                                 latent_dim_size, patch_sizes,
                                                 batch_size, img_size, lsp_layers,
                                                 loss_to_use, training=False)
        return x_hat, kpts, gauss_mask, error_mask, loss

    # TRAINING LOOP
    best_validation_loss, best_validation_epoch = float("inf"), -1
    patience = 0
    step = 0
    for epoch in range(epochs):
        total_nll_loss = 0.0
        total_kl_loss = 0.0
        total_lsp_loss = 0.0
        total_pnet_loss = 0.0
        num_batches = 0
        # TRAIN LOOP
        start_time_epoch = time.time()
        i = 0
        for x_train in train_dataset:

            start_time = time.time()
            loss = train_step(x_train, loss_to_use)

            nll_loss, kl_loss, lsp_loss, pnet_loss = loss[0], loss[1], loss[2], loss[3]

            print("batch number: %4d nll_loss: %4.5f kl_loss: %4.5f lsp_loss: %4.5f pnet_loss: %4.5f took %4.5f s" % (num_batches, nll_loss.numpy(),
                kl_loss.numpy(), lsp_loss.numpy(), pnet_loss.numpy(), time.time() - start_time))

            total_nll_loss = total_nll_loss + nll_loss
            total_kl_loss = total_kl_loss + kl_loss
            total_lsp_loss = total_lsp_loss + lsp_loss
            total_pnet_loss = total_pnet_loss + pnet_loss
            num_batches += 1
            step += 1

            # logging losses to Sacred
            add_sacred_log("train.step_nll_loss", float(nll_loss.numpy()), _run)
            add_sacred_log("train.step_kl_loss", float(kl_loss.numpy()), _run)
            add_sacred_log("train.step_lsp_loss", float(lsp_loss.numpy()), _run)
            add_sacred_log("train.step_pnet_loss", float(pnet_loss.numpy()), _run)
            add_sacred_log("train.step", step, _run)

        epoch_nll_loss = total_nll_loss / num_batches
        epoch_kl_loss = total_kl_loss / num_batches
        epoch_lsp_loss = total_lsp_loss / num_batches
        epoch_pnet_loss = total_pnet_loss / num_batches
        batch_per_second = num_batches / (time.time() - start_time_epoch)

        # logging avg. epoch losses to Sacred
        add_sacred_log("train.epoch_nll_loss", float(epoch_nll_loss.numpy()), _run)
        add_sacred_log("train.epoch_kl_loss", float(epoch_kl_loss.numpy()), _run)
        add_sacred_log("train.epoch_lsp_loss", float(epoch_lsp_loss.numpy()), _run)
        add_sacred_log("train.epoch_pnet_loss", float(epoch_pnet_loss.numpy()), _run)
        add_sacred_log("train.epoch", epoch, _run)

        # VALIDATION LOOP
        # end of every epoch compute validation loss and checkpoint models based on that
        total_valid_nll_loss = 0.0
        total_valid_kl_loss = 0.0
        total_valid_lsp_loss = 0.0
        total_valid_pnet_loss = 0.0
        valid_num_batch = 0
        for x_valid in valid_dataset:
            valid_num_batch = valid_num_batch + 1
            x_pred, kpts, gauss_mask, error_mask, valid_batch_loss = test_step(x_valid)
            val_nll_loss, val_kl_loss, val_lsp_loss, val_pnet_loss = valid_batch_loss[0], \
                        valid_batch_loss[1], valid_batch_loss[2], valid_batch_loss[3]

            total_valid_nll_loss = total_valid_nll_loss + val_nll_loss
            total_valid_kl_loss = total_valid_kl_loss + val_kl_loss
            total_valid_lsp_loss = total_valid_lsp_loss + val_lsp_loss
            total_valid_pnet_loss = total_valid_pnet_loss + val_pnet_loss

        epoch_val_nll_loss = total_valid_nll_loss / valid_num_batch
        epoch_val_kl_loss = total_valid_kl_loss / valid_num_batch
        epoch_val_lsp_loss = total_valid_lsp_loss / valid_num_batch
        epoch_val_pnet_loss = total_valid_pnet_loss / valid_num_batch

        # printing out avg. train  end of every epoch
        print("end of epoch: %2d avg. train_nll_loss: %3.4f avg. train_kl_loss: %3.4f "
            " avg. train_lsp_loss: %3.4f avg. pnet_loss: %3.4f batch/s: %3.4f" % (epoch, epoch_nll_loss.numpy(),
            epoch_kl_loss.numpy(), epoch_val_lsp_loss.numpy(), epoch_val_pnet_loss.numpy(), batch_per_second))

        # logging validation_losses to Sacred
        add_sacred_log("validation.epoch_nll_loss", float(epoch_val_nll_loss.numpy()), _run)
        add_sacred_log("validation.epoch_kl_loss", float(epoch_val_kl_loss.numpy()), _run)
        add_sacred_log("validation.epoch_lsp_loss", float(epoch_val_lsp_loss.numpy()), _run)
        add_sacred_log("validation.epoch_pnet_loss", float(epoch_val_pnet_loss.numpy()), _run)
        add_sacred_log("validation.epoch", epoch, _run)

        # checkpointing models based on validation loss
        validation_loss = epoch_val_pnet_loss + epoch_val_lsp_loss

        if validation_loss < best_validation_loss:
            # update best_validation loss
            best_validation_loss, best_validation_epoch = validation_loss, epoch
            encoder.save_weights(checkpoint_prefix + '_encoder-' + str(best_validation_epoch) + '.h5')
            decoder.save_weights(checkpoint_prefix + '_decoder-' + str(best_validation_epoch) + '.h5')
            for m in range(len(lsp_models)):
                lsp_models[m].save_weights(checkpoint_prefix + '_lsp_model-layer-' + str(lsp_layers[m]) + '-' + str(best_validation_epoch) + '.h5')
            pnet.save_weights(checkpoint_prefix + '_pnet-' + str(best_validation_epoch) + '.h5')
            # early_stopping param resets
            patience = 0
        # early stopping check
        elif validation_loss >= best_validation_loss:
            patience = patience + 1
        # break out if max_patience is reached
        if patience == max_patience:
            break
    print("Training complete!! Best validation loss : %3.4f achieved at epoch"
          ": %2d" % (best_validation_loss, best_validation_epoch))
    add_sacred_log("validation.best_val_loss", float(best_validation_loss), _run)
    add_sacred_log("validation.best_val_epoch", best_validation_epoch, _run)

    return best_validation_loss, best_validation_epoch


@ex.automain
def main(gpu, img_input, noise_type, env, num_keypoints, _config):

    # setup GPU configs
    if not 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # init folder for logging images/checkpoints
    train_dir = os.path.join("permakey_exp", img_input, noise_type, env,
                            str(num_keypoints), datetime.now().isoformat())
    checkpoint_prefix = os.path.join(train_dir, "ckpt")
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
    best_validation_loss, best_validation_epoch = train(checkpoint_prefix=checkpoint_prefix)

    # ckpt load directory
    ckpt_load_dir = train_dir + "/ckpt_"  # ckpt indexed from 1 but epochs from 0

    # run evaluate
    evaluate(ckpt_load_dir=ckpt_load_dir, epoch=best_validation_epoch,
             test_logs_prefix=test_logs_prefix)

    return best_validation_loss

