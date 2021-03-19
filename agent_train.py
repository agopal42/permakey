import os
from pathlib import Path
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import sonnet as snt
import horovod.tensorflow as hvd
from tf_agents import specs
from tf_agents.replay_buffers.episodic_replay_buffer import EpisodicReplayBuffer
from trfl.target_update_ops import update_target_variables
import baselines.common.atari_wrappers as atari_wrappers
from skimage.util import img_as_ubyte
import gym
from sacred import Experiment
import json

# Import agent modules
from rl_loss import vision_forward_pass, encode_keypoints, q_learning
from agent import RecurrentQNet, KptConvEncoder, NodeEncoder, PositionalEncoder, \
	KptGnnEncoder, exploration_policy

# Import keypoint modules
from vision_module import ConvEncoder, Pnet, TransporterEncoder, \
	TransporterDecoder, TransporterKeypointer
from preprocess import add_noise, ENV_NAME_TO_GYM_NAME
from ul_loss import LspMlp, transporter_loss

from utils import add_sacred_log

AUTOTUNE = tf.data.experimental.AUTOTUNE

ex = Experiment()

# init horovod
hvd.init()


@ex.config
def config():
	# Training
	learning_rate = 0.0001
	max_grad_norm = 10
	batch_size = 16
	num_iters = 4*10**5
	learning_starts = 10**4
	train_freq = 4
	env_name = "battlezone"
	n_step_q = 3  # param for n-step q_learning
	double_q = True
	gamma = 0.99  # discount factor for TD-learning
	exp_fraction = 1.0  # fraction of total timesteps after which final_eps is value for eps-greedy exploration
	final_eps = 0.1  # final value for eps-greedy exploration
	eval_eps = 0.0  # evaluation episodes eps-greedy exploration param
	kp_type = "permakey"  # ("transporter", "permakey")
	num_keypoints = 8
	gauss_std = 0.15
	window_size = 8  # window size for recurrent q-learning
	mask_threshold = 0.2
	tau = 0.005  # Polyak avg. constant for target-network update

	# Architecture
	decoder_type = "conv"
	img_input = "dm_atari"  # ("dm_atari" for atari envs)
	img_size = 84  # atari: 84
	colour_input = True
	noise_type = "none"  # ("none", "horizontal", "vertical", "both")
	replay_buffer_size = 350
	kpt_encoder_type = "cnn"  # "cnn" or "gnn"
	kpt_cnn_channels = 32  # number of channels for kpt-conv-encoder
	mp_num_steps = 1  # number of message-passing steps in GNN
	agent_size = 128
	lsp_layers = (0, 1)
	latent_dim_size = 32
	patch_sizes = (2, 2)

	# Logging
	data_dir = "data/atari/"  # atari: data/atari/
	test_every = 10**4
	max_eval_ep = 10  # max. num of episodes used for eval runs
	max_eval_steps = 4000  # max. steps allowed for eval_episode
	record = False  # record eval videos
	vis_ckpt_fname = ""  # directory of ckpts of pre-trained vision model
	vis_load = 0  # ckpt-epoch to load pre-trained keypoint vision model
	gpu = (0,)

	# Eval
	eval_seeds = []
	load_ckpts = []  # ckpt-iter to load agent-ckpts


@ex.named_config
def battlezone():
	# Training
	env_name = "battlezone"
	num_keypoints = 8
	# Architecture
	replay_buffer_size = 500


@ex.named_config
def seaquest():
	# Training
	env_name = "seaquest"
	num_keypoints = 10
	# Architecture
	replay_buffer_size = 1000


@ex.named_config
def mspacman():
	# Training
	env_name = "mspacman"
	num_keypoints = 7
	# Architecture
	replay_buffer_size = 1000


@ex.named_config
def frostbite():
	# Training
	env_name = "frostbite"
	num_keypoints = 20
	# Architecture
	replay_buffer_size = 1200


@ex.capture
def get_optimizer(learning_rate):
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	return optimizer


@ex.capture
def build_vision_model(kp_type, num_keypoints, latent_dim_size, lsp_layers,
					patch_sizes, gauss_std, colour_input):
	model_dict = {}
	if kp_type == "permakey":
		model_dict["encoder"] = ConvEncoder(latent_dim_size)
		model_dict["lsp_model"] = []
		for l in range(len(lsp_layers)):
			output_size = patch_sizes[l]**2*model_dict["encoder"].filters[lsp_layers[l]]
			lsp_net = LspMlp(8*output_size, output_size)
			model_dict["lsp_model"].append(lsp_net)
		model_dict["pnet"] = Pnet(num_keypoints, gauss_std)

	elif kp_type == "transporter":
		model_dict["encoder"] = TransporterEncoder()
		model_dict["keypointer"] = TransporterKeypointer(num_keypoints, gauss_std)
		model_dict["decoder"] = TransporterDecoder(colour=colour_input)
	return model_dict


@ex.capture
def build_agent_model(n_actions, agent_size, batch_size, kpt_encoder_type,
					  kpt_cnn_channels):
	# agent
	model_dict = {"agent_net": RecurrentQNet(agent_size, n_actions, batch_size)}
	# keypoint encoder
	if kpt_encoder_type == "cnn":
		model_dict["kpt_encoder"] = KptConvEncoder(kpt_cnn_channels, agent_size)
	elif kpt_encoder_type == "gnn":
		model_dict["kpt_encoder"] = snt.allow_empty_variables(KptGnnEncoder())
		model_dict["node_enc"] = NodeEncoder(output_dim=agent_size)
		model_dict["pos_net"] = PositionalEncoder(kpt_cnn_channels)
	return model_dict


def load_vision_model(model_dict, kp_type, colour_input, batch_size, lsp_layers,
					patch_sizes, ckpt_load_dir, vis_load):
	# FIX: run forward passes to ensure weight init of encoder and lsp_model
	if kp_type == "permakey":
		if colour_input:
			inputs = tf.zeros((batch_size, 84, 84, 3))
			_, _, _ = model_dict["encoder"](inputs, training=True)
		if not colour_input:
			inputs = tf.zeros((batch_size, 84, 84, 1))

		for l in range(len(lsp_layers)):
			lsp_input = tf.zeros((batch_size, 8 * patch_sizes[l] ** 2 *
								model_dict["encoder"].filters[lsp_layers[l]]))
			_ = model_dict["lsp_model"][l](lsp_input, training=True)
		pnet_inputs = tf.zeros((batch_size, 42, 42, 2))
		_, _ = model_dict["pnet"](pnet_inputs, training=True)
		# load vision module from ckpts
		model_dict["encoder"].load_weights(ckpt_load_dir + 'ckpt_encoder-' + str(vis_load) + '.h5')
		for l in range(len(lsp_layers)):
			model_dict["lsp_model"][l].load_weights(ckpt_load_dir + 'ckpt_lsp_model-layer-' + str(lsp_layers[l]) + '-' + str(vis_load) + '.h5')

		model_dict["pnet"].load_weights(ckpt_load_dir + 'ckpt_pnet-' + str(vis_load) + '.h5')

	elif kp_type == "transporter":
		if colour_input:
			inputs = tf.zeros((batch_size, 84, 84, 3, 2))
			_ = transporter_loss(inputs, model_dict["encoder"], model_dict["keypointer"], model_dict["decoder"], training=True)
		if not colour_input:
			inputs = tf.zeros((batch_size, 84, 84, 1, 2))
			_ = transporter_loss(inputs, model_dict["encoder"], model_dict["keypointer"], model_dict["decoder"], training=True)

		# load vision module from ckpts
		model_dict["encoder"].load_weights(ckpt_load_dir + 'ckpt_encoder-' + str(vis_load) + '.h5')
		model_dict["keypointer"].load_weights(ckpt_load_dir + 'ckpt_keypointer-' + str(vis_load) + '.h5')
		model_dict["decoder"].load_weights(ckpt_load_dir + 'ckpt_decoder-' + str(vis_load) + '.h5')
	return model_dict


@ex.capture
def custom_wrap_deepmind(env, colour_input, episode_life=True, clip_rewards=True):
	"""Configure environment for DeepMind-style Atari."""
	if episode_life:
		env = atari_wrappers.EpisodicLifeEnv(env)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = atari_wrappers.FireResetEnv(env)
	env = atari_wrappers.WarpFrame(env, grayscale=not colour_input)

	if clip_rewards:
		env = atari_wrappers.ClipRewardEnv(env)
	return env


@ex.capture
def make_env(env_name, max_eval_steps, mode, seed=None):
	"""Creates an Atari Gym environment and returns it."""
	env_name = ENV_NAME_TO_GYM_NAME[env_name]  # convert env_name to gym name
	env_seed = seed if mode == "train" else 2*seed
	episode_life = True if mode == "train" else False
	clip_rewards = True if mode == "train" else False

	# create env (using baselines custom wrapper)
	if not mode == "eval":
		print("process seed is %d" % seed)  # TODO(sjoerd): always prints train seed

	env = atari_wrappers.make_atari(env_name, max_episode_steps=max_eval_steps)
	env = custom_wrap_deepmind(
		env, episode_life=episode_life, clip_rewards=clip_rewards)
	# seeding env and action space
	env.seed(env_seed)
	env.action_space.seed(env_seed)
	return env


def update_target_networks(source_model_dict, target_model_dict, tau):
	"""
	Helper function to perform target network updates
	:param source_nets: (list) of source networks tf.keras.Module type
	:param target_nets: (list) of target networks
	:param tau: (float) Polyak weight avg. param
	:return:
	"""
	# perform Polyak avg. i.e. "soft" updates
	source_vars, target_vars = [], []
	for model in source_model_dict.keys():
		source_vars = source_vars + list(source_model_dict[model].trainable_variables)
		target_vars = target_vars + list(target_model_dict[model].trainable_variables)

	# updating target networks
	update_target_variables(target_vars, source_vars, tau)
	return source_vars, target_vars


@ex.capture
def eval_step(eval_env, vision_model_dict, agent_model_dict, eval_eps, max_eval_ep,
	agent_size, lsp_layers, kp_type, mask_threshold, patch_sizes, img_size,
	kpt_encoder_type, noise_type, mp_num_steps):

	# Run max_eval_ep number of episodes using greedy-policy inferred
	# from q-function and compute avg. episodic reward
	eval_ep_rewards = [0.0]
	obs = eval_env.reset()
	reset = True
	num_ep = 0
	eval_c_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
	eval_h_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)

	while num_ep < max_eval_ep:
		obs_float = np.asarray(obs[None, :, :, :], dtype=np.float32) / 255.0
		# sometimes add distractors
		if noise_type != "none":
			obs_float = add_noise(obs_float[0, :, :, :], noise_type)
			obs_float = obs_float[None, :, :, :]
		# vision-module forward pass
		bottom_up_maps, encoder_features, kpt_centers = vision_forward_pass(
			tf.constant(obs_float), vision_model_dict, lsp_layers, kp_type,
			patch_sizes, img_size)

		# compute keypoint encodings
		bottom_up_features = encode_keypoints(bottom_up_maps, encoder_features,
			kpt_centers, mask_threshold, kp_type, kpt_encoder_type,
			mp_num_steps, q_learn=False, pos_net=agent_model_dict.get("pos_net"),
			node_encoder=agent_model_dict.get("node_enc"),
			kpt_encoder=agent_model_dict.get("kpt_encoder")
		)  # passes None if not available

		# agent step
		action, eval_h_t, eval_c_t = agent_model_dict["agent_net"].step(
			bottom_up_features, [eval_h_tm1, eval_c_tm1], eval_eps,
			training=False, stochastic=True)
		# env step
		new_obs, rew, done, _ = eval_env.step(action)
		eval_ep_rewards[-1] += rew
		obs = new_obs
		eval_h_tm1, eval_c_tm1 = eval_h_t, eval_c_t
		# episode termination
		if done:
			obs = eval_env.reset()
			# reset lstm cell state at episode end
			eval_c_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
			eval_h_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
			num_ep = num_ep + 1
			# if hvd.local_rank() == 0:
			# 	print(eval_ep_rewards[-1])
			eval_ep_rewards.append(0.0)
			reset = True

	# log episodic return stats
	avg_eval_ep_return = np.mean(np.array(eval_ep_rewards[0:-1]), axis=0)
	std_ep_return = np.std(np.array(eval_ep_rewards[0:-1]), axis=0)
	min_ep_return = np.amin(np.array(eval_ep_rewards[0:-1]), axis=0)
	max_ep_return = np.amax(np.array(eval_ep_rewards[0:-1]), axis=0)
	return avg_eval_ep_return, std_ep_return, min_ep_return, max_ep_return


@ex.capture
def load_ckpt_info(env_name, kp_type, img_input, noise_type,
				   num_keypoints, vis_ckpt_fname):

	# paths to saved vision_module weights for diff. envs and kpt_types
	ckpt_load_dir = kp_type + "_exp/" + img_input + "/" + noise_type + "/" + \
					env_name + "/" + str(num_keypoints) + "/" + vis_ckpt_fname + "/"

	if not os.path.exists(ckpt_load_dir):
		print("pre-trained vision model ckpt does not exist!!!")
	return ckpt_load_dir


@ex.capture
def train(gamma, double_q, n_step_q, exp_fraction, final_eps, kp_type, colour_input,
		patch_sizes, lsp_layers, batch_size, num_iters, learning_starts, train_freq,
		kpt_encoder_type, kpt_cnn_channels, agent_size, learning_rate, max_grad_norm,
		mask_threshold, tau, window_size, ckpts_prefix, ckpt_load_dir, vis_load,
		test_every, mp_num_steps, img_size, replay_buffer_size, seed, noise_type,
		_run):

	model_init_start = time.time()
	process_seed = seed + hvd.local_rank()

	# init Gym environments
	train_env = make_env(mode="train", seed=process_seed)
	if hvd.local_rank() == 0:  # eval only on 1 node (horovod)
		eval_env = make_env(mode="eval", seed=20*(process_seed+1))
	n_actions = train_env.action_space.n

	# build models
	vision_model_dict = build_vision_model()
	agent_model_dict = build_agent_model(n_actions=n_actions,
										 kpt_cnn_channels=kpt_cnn_channels)
	target_agent_model_dict = build_agent_model(n_actions=n_actions,
												kpt_cnn_channels=kpt_cnn_channels)

	# Horovod: adjust learning rate based on number of GPUs.
	optimizer = get_optimizer(learning_rate=learning_rate * hvd.size())

	# setting up ckpts for all the modules
	query_ckpt, attn_ckpt, pos_enc_ckpt, node_enc_ckpt, \
	scene_ckpt, kpt_enc_ckpt = None, None, None, None, None, None

	policy_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["agent_net"])

	kpt_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["kpt_encoder"])
	if kpt_encoder_type == "gnn":
		node_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["node_enc"])
		pos_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["pos_net"])

	# load pre-trained vision module
	vision_model_dict = load_vision_model(vision_model_dict, kp_type,
										colour_input, batch_size, lsp_layers,
										patch_sizes, ckpt_load_dir, vis_load)

	if hvd.local_rank() == 0:
		print("initializing models and env took %4.5f s" % (time.time() - model_init_start))

	def train_step(inputs):
		# Minimize the TD error on a batch sampled from replay buffer.
		with tf.GradientTape() as tape:
			step_loss, extra = q_learning(vision_model_dict, agent_model_dict,
				target_agent_model_dict, inputs, batch_size, kp_type, agent_size,
				mask_threshold, patch_sizes, kpt_encoder_type, mp_num_steps,
				img_size, lsp_layers, window_size, gamma, double_q, n_step_q)
		w_update_start = time.time()
		# Horovod: add Horovod Distributed GradientTape.
		tape = hvd.DistributedGradientTape(tape)

		# collecting trainable params of all modules
		params = []
		for agent_model in agent_model_dict.values():
			params = params + list(agent_model.trainable_variables)

		# compute grads
		grads = tape.gradient(step_loss, params)
		# apply grad clipping
		grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=max_grad_norm)
		# update agent
		optimizer.apply_gradients(zip(grads, params))
		# print("grad comp + weight updates take %4.5f" % (time.time() - w_update_start))
		return step_loss, extra

	# load weights using var assignment
	source_vars, target_vars = update_target_networks(agent_model_dict,
												target_agent_model_dict, tau)

	# init replay buffer
	data_spec = (specs.TensorSpec([84, 84, 3], tf.int32, 'obs_tm1'),
				 specs.TensorSpec([1], tf.int32, 'a_tm1'),
				 specs.TensorSpec([1], tf.float32, 'r_tm1'),
				 specs.TensorSpec([2], tf.float32, 'begin_end'))
	# each process has it's own smaller reply_buffer
	replay_buffer = EpisodicReplayBuffer(capacity=int(replay_buffer_size),
										buffer_size=8,
										dataset_drop_remainder=False,
										data_spec=data_spec,
										begin_episode_fn=lambda x: bool(x[3][0, 0]),
										end_episode_fn=lambda x: bool(x[3][0, 1]))

	# create tf.Dataset object from replay_buffer and sample
	rb_ds = replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=window_size+n_step_q+1)

	# dataset iterator sampling trajectories from replay_buffer
	episode_ids = replay_buffer.create_episode_ids(1)
	rb_ds = rb_ds.prefetch(buffer_size=AUTOTUNE)
	rb_iterator = iter(rb_ds)

	episode_rewards = [0.0]
	obs = train_env.reset()
	reset = False

	# lists for logging exp results
	eps = 0.1
	episode_timestep = 0
	exploration = exploration_policy(num_iters, exp_fraction, final_eps)
	avg_td_error = 0.0
	# init lstm_agent state
	c_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
	h_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
	best_eval_score = -float("inf")

	# TRAINING LOOP
	for t in range(int(num_iters)):
		# Horovod: broadcast initial variable states from rank 0 to all other processes.
		# This is necessary to ensure consistent initialization of all workers when
		# training is started with random weights or restored from a checkpoint.
		if t == 0:
			hvd.broadcast_variables(source_vars, root_rank=0)
			hvd.broadcast_variables(target_vars, root_rank=0)
			hvd.broadcast_variables(optimizer.variables(), root_rank=0)

		online_step_start = time.time()

		# convert obs to float and scale to 0-1
		obs_float = np.asarray(obs[None, :, :, :], dtype=np.float32) / 255.0
		# sometimes add distractors
		if noise_type is not "none":
			obs_float = add_noise(obs_float[0, :, :, :], noise_type)
			obs_float = obs_float[None, :, :, :]
		# exploration
		update_eps = tf.constant(exploration.value(t))

		# compute forward pass of input obs over vision + attention modules
		bottom_up_masks, encoder_features, kpt_centers = vision_forward_pass(
			obs_float, vision_model_dict, lsp_layers, kp_type,
			patch_sizes, img_size)

		# compute keypoint encodings

		bottom_up_features = encode_keypoints(bottom_up_masks, encoder_features,
			kpt_centers, mask_threshold, kp_type, kpt_encoder_type,
			mp_num_steps, q_learn=False, pos_net=agent_model_dict.get("pos_net"),
			node_encoder=agent_model_dict.get("node_enc"),
			kpt_encoder=agent_model_dict.get("kpt_encoder"))  # passes None if not available

		# agent step
		action, h_t, c_t = agent_model_dict["agent_net"].step(bottom_up_features,
						[h_tm1, c_tm1], update_eps, training=True, stochastic=True)
		# env step
		new_obs, rew, done, _ = train_env.step(action)

		episode_timestep = episode_timestep + 1
		episode_rewards[-1] += rew

		# store transitions in replay buffer
		store_exp_start = time.time()
		# making data_tuple compatible for add_batch() method
		obs = img_as_ubyte(np.array(obs_float[0, :, :, :], dtype=float))
		action = np.array(action, dtype=np.int32)
		rew = np.array(rew, ndmin=1, dtype=np.float32)
		end = np.array(done, ndmin=1, dtype=np.float32)
		begin = np.array(reset, ndmin=1, dtype=np.float32)
		begin_end = np.concatenate((begin, end), axis=0)
		# converting from
		values = (obs, action, rew, begin_end)
		values_batched = tf.nest.map_structure(lambda b: tf.stack([b]), values)
		# add batch of transitions of episode_ids to replay_buffer
		episode_ids = replay_buffer.add_batch(values_batched, episode_ids)

		obs = new_obs
		h_tm1 = h_t
		c_tm1 = c_t
		reset = False
		# episode termination
		if done:
			# saving cummulative returns at end of episode
			print("Episode Return: %3.3f" % (episode_rewards[-1]))
			print(episode_ids.numpy(), update_eps.numpy())
			obs = train_env.reset()
			episode_timestep = 0
			# reset lstm state at episode end
			c_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
			h_tm1 = tf.Variable(tf.zeros((1, agent_size)), trainable=False)
			episode_rewards.append(0.0)
			reset = True

		# Q_LEARNING UPDATES BEGIN
		if t > learning_starts and t % train_freq == 0:
			batch_q_start = time.time()
			# sample a batch of trajectories from replay_buffer for recurrent-dqn
			inputs = next(rb_iterator)
			step_loss, extra = train_step(inputs)
			step_loss = hvd.allreduce(step_loss)

			# soft-update target networks
			update_start = time.time()
			source_vars, target_vars = update_target_networks(
				agent_model_dict, target_agent_model_dict, tau)
			# print("Target network updates take %4.5f" % (time.time() - update_start))
			td_error = tf.reduce_mean(hvd.allreduce(extra.td_error), axis=0)

			if hvd.local_rank() == 0:
				print("Iteration: %5d Step loss: %4.4f, TD_error: %3.4f took %4.5f s" %
					  (t, step_loss, td_error, time.time() - batch_q_start))

				# logging step losses to sacred
				add_sacred_log("train.t", int((t-learning_starts)/train_freq), _run)
				add_sacred_log("train.step_loss", float(step_loss), _run)
				add_sacred_log("train.step_td_error", float(td_error), _run)

			avg_td_error = avg_td_error + np.abs(td_error)
		# VALIDATION/CKPT
		if t > learning_starts and t % test_every == 0:
			# trigger evaluation run on only 1 node
			if hvd.local_rank() == 0:
				eval_start = time.time()
				mean_ep_rew, var_ep_rew, _, _ = eval_step(eval_env, vision_model_dict,
								agent_model_dict)
				avg_td_error = avg_td_error / float((t - learning_starts) / train_freq)

				print("Evaluation after: %5d steps avg_ep_return: %4.5f running_avg_td_error: %4.5f took %4.5f s" %
					(t, mean_ep_rew, avg_td_error, time.time() - eval_start))

				# logging avg. episodic rewards to sacred
				add_sacred_log("test.t", int((t-learning_starts)/train_freq), _run)
				add_sacred_log("test.mean_ep_return", float(mean_ep_rew), _run)
				add_sacred_log("test.var_ep_return", float(var_ep_rew), _run)
				add_sacred_log("test.avg_td_error", float(avg_td_error), _run)

				avg_td_error = 0.0

				# ckpt model based on eval-run scores
				if mean_ep_rew > 0.95*best_eval_score:
					best_eval_score = mean_ep_rew
					# Horovod: save checkpoints only on worker 0 to prevent other workers from
					# corrupting it.
					policy_ckpt.save(ckpts_prefix + '_agent_net')
					kpt_enc_ckpt.save(ckpts_prefix + '_kpt_encoder')
					if kpt_encoder_type == "gnn":
						node_enc_ckpt.save(ckpts_prefix + '_node_enc')
						pos_enc_ckpt.save(ckpts_prefix + '_pos_net')

	if hvd.local_rank() == 0:
		print("Training complete!!!")


@ex.command
def evaluate(env_name, kp_type, colour_input, batch_size, lsp_layers, num_keypoints,
			learning_rate, patch_sizes, kpt_encoder_type, kpt_cnn_channels,
			noise_type, record, eval_seeds, img_input, vis_ckpt_fname, vis_load,
			load_ckpts, _config, _run):

	# setting up visible GPU device list
	gpus = tf.config.experimental.list_physical_devices('GPU')
	visible_gpus = [gpus[vis_idx] for vis_idx in _config["gpu"]]
	for gpu in visible_gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	# Horovod: pin GPU to be used to process local rank (one GPU per process)
	if visible_gpus:
		tf.config.experimental.set_visible_devices(visible_gpus[hvd.local_rank()], 'GPU')

	# list of seeds/policies to evaluate
	eval_dir = os.path.join("rl_exp", img_input, noise_type, kp_type,
				kpt_encoder_type, env_name)
	eval_models = sorted(Path(eval_dir).iterdir())
	eval_models = list(map(str, eval_models))
	eval_env_seeds = eval_seeds
	vis_ckpt_load_dir = load_ckpt_info(env_name, kp_type, img_input, noise_type,
									num_keypoints, vis_ckpt_fname)
	# init dummy env just for n_actions var
	dummy_env = make_env(mode="eval", seed=666)
	n_actions = dummy_env.action_space.n
	dummy_env.close()
	# load vision modules
	vis_model_dict = build_vision_model()
	vis_model_dict = load_vision_model(vis_model_dict, kp_type,
						colour_input, batch_size, lsp_layers,
						patch_sizes, vis_ckpt_load_dir, vis_load)
	# load agent modules
	agent_model_dict = build_agent_model(n_actions=n_actions,
										 kpt_cnn_channels=kpt_cnn_channels)
	# init ckpts for agent modules
	# setting up ckpts for all the modules
	optimizer = get_optimizer(learning_rate=learning_rate * hvd.size())

	# setting up ckpts for all the modules
	query_ckpt, attn_ckpt, pos_enc_ckpt, node_enc_ckpt, scene_ckpt, \
	kpt_enc_ckpt = None, None, None, None, None, None

	policy_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["agent_net"])
	kpt_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["kpt_encoder"])
	if kpt_encoder_type == "gnn":
		node_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["node_enc"])
		pos_enc_ckpt = tf.train.Checkpoint(optimizer=optimizer,
									model=agent_model_dict["pos_net"])

	# Restore agent module ckpts
	eval_runs_scores = []
	for eval_env_seed in eval_env_seeds:
		aggregate_mean_rew, aggregate_std_rew = 0.0, 0.0
		# init eval env
		print("Initializing eval env with seed %d" % (eval_env_seed))
		eval_env = make_env(mode="eval", seed=eval_env_seed)
		if record:
			eval_env = gym.wrappers.Monitor(eval_env, "videos/",
											video_callable=lambda x: True,
											force=True)
		mean_ep_rew, std_ep_rew = [], []
		for eval_model in range(len(eval_models)):
			# setup logging dir
			# test_logs_prefix_dir = os.path.join(eval_models[eval_model], "eval_logs")
			# video_dir = os.path.join(eval_models[i], "videos/")

			# load ckpt agent modules of ith policy
			policy_ckpt.restore(eval_models[eval_model]+'/'+'ckpt_agent_net-'+
								str(load_ckpts[eval_model])).expect_partial()
			kpt_enc_ckpt.restore(eval_models[eval_model]+'/'+'ckpt_kpt_encoder-'+
								str(load_ckpts[eval_model])).expect_partial()
			if kpt_encoder_type == "gnn":
				pos_enc_ckpt.restore(eval_models[eval_model]+'/'+'ckpt_pos_net-'+
								str(load_ckpts[eval_model])).expect_partial()
				node_enc_ckpt.restore(eval_models[eval_model]+'/'+'ckpt_node_enc-'+
								str(load_ckpts[eval_model])).expect_partial()

			# run evaluation
			ep_mu, ep_std, ep_min, ep_max = eval_step(eval_env, vis_model_dict,
													agent_model_dict)
			mean_ep_rew.append(ep_mu)
			std_ep_rew.append(ep_std)
			print("Min/Max reward is %5.2f/%5.2f for policy %d" %
				(ep_min, ep_max, eval_model))
		# compute aggregated rew_mu, rew_sigma over all samples
		aggregate_mean_rew = np.mean(np.array(mean_ep_rew))
		aggregate_std_rew = np.sqrt(np.sum(np.square(np.array(std_ep_rew))) /
									len(eval_models))
		print("Mean reward: %4.5f std_dev: %4.5f over all policies "
			"for eval-env with seed %d" % (aggregate_mean_rew,
			aggregate_std_rew, eval_env_seed))
		eval_runs_scores.append((aggregate_mean_rew, aggregate_std_rew))

	median_run_mu = np.median(np.array(list(zip(*eval_runs_scores))[0]))
	median_run_idx = list(zip(*eval_runs_scores))[0].index(median_run_mu)
	print("Median eval run scores are mu: %4.5f std-dev: %4.5f" %
		  (median_run_mu, eval_runs_scores[median_run_idx][1]))
	return median_run_mu, eval_runs_scores[median_run_idx][1]


@ex.automain
def main(env_name, noise_type, kp_type, num_keypoints, img_input,
		 kpt_encoder_type, replay_buffer_size, vis_ckpt_fname, _config):

	# setting up visible GPU device list
	gpus = tf.config.experimental.list_physical_devices('GPU')
	visible_gpus = [gpus[vis_idx] for vis_idx in _config["gpu"]]
	for gpu in visible_gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	# Horovod: pin GPU to be used to process local rank (one GPU per process)
	if visible_gpus:
		tf.config.experimental.set_visible_devices(visible_gpus[hvd.local_rank()], 'GPU')

	# init folder for logging rl_exp stuff
	train_dir = os.path.join("rl_exp", img_input, noise_type, kp_type,
				kpt_encoder_type, env_name, datetime.now().isoformat())
	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
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

	ckpt_load_dir = load_ckpt_info(env_name, kp_type, img_input, noise_type,
								   num_keypoints, vis_ckpt_fname)
	# TRAIN AGENT
	train(replay_buffer_size=replay_buffer_size, ckpts_prefix=ckpt_prefix,
	ckpt_load_dir=ckpt_load_dir)
	return 0
