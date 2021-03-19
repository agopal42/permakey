import tensorflow as tf
import numpy as np
import time
import trfl
from graph_nets import graphs

# Import keypoint modules
from ul_loss import transporter_loss, lsp_loss


def get_graph_tuple(nodes, globals=None):
	"""
	Helper function to create a dict w/ relevant fields for graph_data
	:param nodes: (Tensor) batch of node values (batch, num_kpts, node_feature_dims)
	:param globals: (Tensor) batch of global values (batch, 1, global_feature_dims)
	:return:
		graph_tuple: graph.GraphTuple type
	"""
	nodes_shape = nodes.shape
	batch_size = nodes_shape[0]
	num_nodes = tf.ones([batch_size], dtype=tf.int32)
	num_edges = tf.ones([batch_size], dtype=tf.int32)
	# defining num_nodes & num_edges for each sample in a batch of input graphs
	b_num_nodes = nodes_shape[1]*num_nodes
	b_num_edges = (nodes_shape[1]**2)*num_edges
	# rehaping (b, num_nodes, dims) -> (b*num_nodes, dims)
	nodes = tf.reshape(nodes, [nodes_shape[0] * nodes_shape[1], nodes_shape[2]])
	if globals is not None:
		globals_shape = globals.shape
		globals = tf.reshape(globals, [globals_shape[0] * globals_shape[1],
							globals_shape[2]])
		graph_tuple = graphs.GraphsTuple(nodes=nodes, globals=globals)
	else:
		graph_tuple = graphs.GraphsTuple(nodes=nodes, globals=None, edges=None,
										 n_node=b_num_nodes, n_edge=b_num_edges,
										 senders=None, receivers=None)
	return graph_tuple


def collect_keypoint_features(encoder_activations, bottom_up_maps, threshold,
	kp_type):
	"""
	:param encoder_activations = highest (LSP) layer of conv_encoder activations
	:param bottom_up_maps = gauss windows around keypoints (batch, H, W, num_keypoints)
	:param threshold = (float) value to decide mask-on keypoint activations
	:return:
	keypoint_features: (batch, features, num_keypoints)
	"""

	if kp_type == "permakey":
		enc_shape = encoder_activations.shape
		bottom_up_maps = tf.image.resize(bottom_up_maps, (enc_shape[1], enc_shape[2]), method='nearest')

	act_shape = encoder_activations.shape
	n_kpts = bottom_up_maps.shape[3]

	# binarize heatmaps based on threshold
	binary_maps = tf.where(tf.math.greater_equal(bottom_up_maps, threshold), tf.constant(1.0), tf.constant(0.0))
	# (batch, H, W, n_kpts) -> (batch, H, W, 1, n_kpts)
	binary_maps = tf.transpose(tf.expand_dims(binary_maps, axis=4), perm=[0, 1, 2, 4, 3])
	nnz_counts = tf.cast(tf.math.count_nonzero(binary_maps, axis=[1, 2]), dtype=tf.float32)
	# compute mask-on activations
	enc_act_stacked = tf.stack([encoder_activations]*n_kpts, axis=4)
	mask_on_activations = tf.math.multiply(enc_act_stacked, binary_maps)
	# avg mask_on_activations spatially to compute keypoint_features
	keypoint_features = tf.math.reduce_sum(mask_on_activations, axis=[1, 2]) / nnz_counts
	keypoint_features = tf.transpose(keypoint_features, perm=[0, 2, 1])
	return keypoint_features


def vision_forward_pass(inputs, vision_model_dict, lsp_layers, kp_type,
						patch_sizes, img_size):
	"""
	:param inputs: input images (greyscale or colour) for vision_module (batch*timesteps, H, W, C)
	:param vision_model_dict: (dict) of vision networks {"encoder", "lsp_model", "pnet"}
								or {"encoder", "keypointer", "decoder"}
	:param lsp_layers: (list) selected layers for lsp computation
	:param kp_type: "transporter" or "permakey" type of keypoint method used
	:param patch_sizes: (tuple)
	:param img_size: (int) size of input images
	:return:
	mask = botom-up (un)-predictability heatmaps (batch_size, H, W, num_kpts)
	encoder_activations = (b, H, W, C)
	kpts = keypoint locations (b, num_keypoints, 2)
	"""
	# global variables
	encoder_activations, bottom_up_map, kpts = 0.0, 0.0, 0.0
	if kp_type == "permakey":
		mu, var, encoder_activations = vision_model_dict["encoder"](inputs,
																	training=False)
		# run lsp on activation patches
		kpts, bottom_up_map, stacked_error_masks, _, _ = lsp_loss(
			vision_model_dict["lsp_model"], encoder_activations, patch_sizes,
			img_size, lsp_layers, pnet=vision_model_dict["pnet"], training=False)
		# storing only last lsp_layer activations
		encoder_activations = encoder_activations[lsp_layers[-1]]
	elif kp_type == "transporter":
		inputs = tf.stack([inputs, inputs], axis=4)
		kpts, bottom_up_map, encoder_activations, pred, _ = transporter_loss(
			inputs, vision_model_dict["encoder"], vision_model_dict["keypointer"],
			vision_model_dict["decoder"], training=False)
	# collect_glimpse_start = time.time()
	return bottom_up_map, encoder_activations, kpts


def encode_keypoints(bottom_up_map, encoder_activations, kpt_centers, mask_threshold,
					kp_type, kpt_encoder_type, mp_num_steps, q_learn,
					kpt_encoder=None, node_encoder=None, pos_net=None):
	"""
	Function to that encodes features of pre-trained keypoints using either
	1. GNN (ours) or 2. CNN (Transporter)
	:param bottom_up_map: (Tensor) [b, T, H, W, K] keypoint masks
	:param encoder_activations: (Tensor) [b, T, H, W, C]
	:param kpt_centers: (Tensor) [b, K, 2] keypoint centers
	:param mask_threshold: (float) threshold value to binarize keypoint masks
	:param kp_type: (str) "transporter" or "permakey"
	:param mp_num_steps: (int) number of message-passing steps for GNN
	:param kpt_encoder: keypoint feature encoder network (cnn or gnn)
	:param pos_net: positional encoding network (mlp)
	:param q_learn: (bool) True for q_updates q_learning False for online steps
	:return:
	"""
	scene_z, bottom_up_features = 0.0, 0.0

	# KeyQN (Transporter-style) processing
	if kpt_encoder_type == "cnn":
		# heatmaps * features
		heatmaps = tf.reduce_sum(bottom_up_map, axis=3, keepdims=True)
		# binarize keypoint maps
		binary_heatmaps = tf.where(tf.math.greater_equal(heatmaps, mask_threshold),
											tf.constant(1.0), tf.constant(0.0))
		# resize maps if "permakey" used
		masked_activations = tf.math.multiply(encoder_activations, binary_heatmaps)  # layer_1 activations
		# pass masked activations through cnn to encode keypoints features
		bottom_up_features = kpt_encoder(masked_activations, training=q_learn)
	# GNN set-based processing of keypoints (ours)
	elif kpt_encoder_type == "gnn":
		# collect features at keypoint locations
		kpt_features = collect_keypoint_features(encoder_activations, bottom_up_map,
												mask_threshold, kp_type)
		# compute learned positional embedding
		pos_encoding = pos_net(kpt_centers, training=q_learn)
		# append positional encodings to kpt_features
		bottom_up_features = tf.concat([kpt_features, pos_encoding], axis=2)
		b_size, num_nodes = bottom_up_features.shape[0:2]
		output_graphs = kpt_encoder(bottom_up_features, mp_num_steps, is_training=q_learn)
		# node encoding
		nodes = output_graphs.nodes
		nodes = tf.reshape(nodes, [b_size, -1])
		# apply MLP encoder to concat output node_values
		bottom_up_features = node_encoder(nodes, training=q_learn)
	return bottom_up_features


def q_learning(vision_model_dict, agent_model_dict, target_agent_model_dict,
			inputs, batch_size, kp_type, agent_size, mask_threshold,
			patch_sizes, kpt_encoder_type, mp_steps, img_size, lsp_layers,
			window_size, gamma, double_q, n_step_q):
	"""
	:param vision_model_dict:
	:param agent_model_dict:
	:param target_agent_model_dict:
	:param inputs: bottom_up_kpt inputs [batch, T, dims]
	:param batch_size: (int)
	:param kp_type: (str) "transporter" or "permakey" type of keypoint used for bottom-up processing
	:param agent_size: (int) size of agent lstm
	:param mask_threshold: (float)
	:param patch_sizes: (int) size of patch size for "permakey" keypoints
	:param kpt_encoder_type: (str) "cnn" for conv-net "gnn" for graph-net
	:param mp_steps: (int) number of message-passing steps in GNNs
	:param img_size: (int) size of input image (H for H x H img)
	:param lsp_layers: (tuple) of layers for "permakey" keypoints
	:param window_size: (int) size of window used for recurrent q-learning
	:param gamma: (float) discount factor
	:param double_q: (bool) True if using double q-learning
	:param n_step_q: (int) 'n' value used for n-step q-learning
	:return:
	bottom_up_maps: keypoint gaussian masks
	bottom_up_features: bottom-up keypoint features
	"""

	# unpacking elements from sampled trajectories from buffer
	obses_tm1, a_tm1, r_t, dones = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3]

	obses_tm1 = tf.cast(obses_tm1, dtype=tf.float32)/255.0  # (batch, T, H, W)

	# reshaping obs tensor (batch, T, H, W, C) -> (batch*T, H, W, C)
	obses_tm1_shape = obses_tm1.shape
	obses_tm1 = tf.reshape(obses_tm1, [obses_tm1_shape[0] * obses_tm1_shape[1],
						obses_tm1_shape[2], obses_tm1_shape[3], obses_tm1_shape[4]])

	# 1 single forward pass of kpt-module for T-steps of frames
	vis_forward_start = time.time()
	bottom_up_maps, encoder_features, kpt_centers = vision_forward_pass(
		obses_tm1, vision_model_dict, lsp_layers, kp_type, patch_sizes, img_size)

	# reshaping tensors from (b*T, ...) -> (b, T, ...)
	bup_map_shape = bottom_up_maps.shape
	bottom_up_maps = tf.reshape(bottom_up_maps, [obses_tm1_shape[0], obses_tm1_shape[1],
								bup_map_shape[1], bup_map_shape[2], bup_map_shape[3]])
	enc_feat_shape = encoder_features.shape
	encoder_features = tf.reshape(encoder_features, [obses_tm1_shape[0], obses_tm1_shape[1],
								enc_feat_shape[1], enc_feat_shape[2], enc_feat_shape[3]])
	kpt_c_shape = kpt_centers.shape
	kpt_centers = tf.reshape(kpt_centers, [obses_tm1_shape[0], obses_tm1_shape[1],
							kpt_c_shape[1], kpt_c_shape[2]])

	# splitting outputs into 2 parts  targets = (1:T) and qs = (0:T-1)
	bottom_up_maps_tm1, bottom_up_maps_t = bottom_up_maps[:, n_step_q:-1, :, :, :], bottom_up_maps[:, n_step_q+1:, :, :, :]
	encoder_features_tm1, encoder_features_t = encoder_features[:, n_step_q:-1, :, :, :], encoder_features[:, n_step_q+1:, :, :, :]
	kpt_centers_tm1, kpt_centers_t = kpt_centers[:, n_step_q:-1, :, :], kpt_centers[:, n_step_q+1:, :, :]

	# collecting a_tm1, r_t and dones for n'th step bootstrapping
	a_tm1, r_t = tf.cast(a_tm1, dtype=tf.int32), tf.cast(r_t, dtype=tf.float32)
	a_tm1, r_t = a_tm1[:, n_step_q:-1, :], r_t[:, 0:-1, :]
	dones = tf.cast(dones, dtype=tf.float32)
	dones = dones[:, n_step_q+1:, 1]  # dones for q_t's
	# switching batch and time axis to align all inputs i.e. (T, b, ..) -> (b, T, ..)
	a_tm1 = tf.transpose(a_tm1, perm=[1, 0, 2])
	dones = tf.transpose(dones, perm=[1, 0])

	# reshaping tensors again (ugh!) (b, T-1, ...) -> (b*(T-1), ...)
	bup_tm1_shape = bottom_up_maps_tm1.shape
	bottom_up_maps_tm1 = tf.reshape(bottom_up_maps_tm1, [-1, bup_tm1_shape[2],
						bup_tm1_shape[3], bup_tm1_shape[4]])
	bottom_up_maps_t = tf.reshape(bottom_up_maps_t, bottom_up_maps_tm1.shape)

	enc_tm1_shape = encoder_features_tm1.shape
	encoder_features_tm1 = tf.reshape(encoder_features_tm1, [-1, enc_tm1_shape[2],
							enc_tm1_shape[3], enc_tm1_shape[4]])
	encoder_features_t = tf.reshape(encoder_features_t, encoder_features_tm1.shape)

	kptc_tm1_shape = kpt_centers_tm1.shape
	kpt_centers_tm1 = tf.reshape(kpt_centers_tm1, [-1, kptc_tm1_shape[2], kptc_tm1_shape[3]])
	kpt_centers_t = tf.reshape(kpt_centers_t, kpt_centers_tm1.shape)

	# compute keypoint encodings
	kpts_features_tm1 = encode_keypoints(bottom_up_maps_tm1, encoder_features_tm1,
							kpt_centers_tm1, mask_threshold, kp_type, kpt_encoder_type,
							mp_steps, True, pos_net=agent_model_dict.get("pos_net"),
							kpt_encoder=agent_model_dict.get("kpt_encoder"),
							node_encoder=agent_model_dict.get("node_enc"))  # passes none if not available

	kpts_features_t = encode_keypoints(bottom_up_maps_t, encoder_features_t,
							kpt_centers_t, mask_threshold, kp_type, kpt_encoder_type,
							mp_steps, True, pos_net=target_agent_model_dict.get("pos_net"),
							kpt_encoder=target_agent_model_dict.get("kpt_encoder"),
							node_encoder=target_agent_model_dict.get("node_enc")) # passes none if not available

	# reshaping back the time axis (b*T, dims) -> (b, T, dims)
	kpts_features_tm1 = tf.expand_dims(kpts_features_tm1, axis=1)
	kpts_tm1_shape = kpts_features_tm1.shape
	kpts_features_tm1 = tf.reshape(kpts_features_tm1, [batch_size, window_size,
								kpts_tm1_shape[-1]])

	kpts_features_t = tf.expand_dims(kpts_features_t, axis=1)
	kpts_t_shape = kpts_features_t.shape
	kpts_features_t = tf.reshape(kpts_features_t, [batch_size, window_size,
								kpts_t_shape[-1]])

	# RNN computation
	q_tm1_seq = []
	q_t_seq = []
	q_t_selector_seq = []

	# reset lstm state at start of update as in R-DQN random updates
	c_tm1 = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=True)
	h_tm1 = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=True)
	h_t_sel = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=True)
	c_t_sel = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=True)
	h_t = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=False)  # td_targets
	c_t = tf.Variable(tf.zeros((batch_size, agent_size)), trainable=False)  # td_targets
	rnn_unroll_start = time.time()

	# RNN unrolling
	for seq_idx in tf.range(window_size):
		s_tm1 = kpts_features_tm1[:, seq_idx, :]
		s_t = kpts_features_t[:, seq_idx, :]
		# double_q action selection step
		if double_q:
			q_t_selector, h_t_sel, c_t_sel = agent_model_dict["agent_net"](s_t, [h_t_sel, c_t_sel], training=True)
			q_t_selector_seq.append(q_t_selector)

		q_tm1, h_tm1, c_tm1 = agent_model_dict["agent_net"](s_tm1, [h_tm1, c_tm1], training=True)
		q_tm1_seq.append(q_tm1)
		q_t, h_t, c_t = target_agent_model_dict["agent_net"](s_t, [h_t, c_t], training=False)
		q_t_seq.append(q_t)
	# print("RNN for loop unrolling took %s" % (time.time() - rnn_unroll_start))

	q_tm1 = tf.convert_to_tensor(q_tm1_seq, dtype=tf.float32)
	q_t = tf.convert_to_tensor(q_t_seq, dtype=tf.float32)

	# compute cumm. rew for 'n' steps
	if n_step_q > 1:
		l = tf.constant(np.array(list(range(n_step_q))), dtype=tf.float32)
		discounts = tf.math.pow(gamma, l)
		# slice r_t [b, T] into moving windows of [b, t-k, k]  # cumsum over k steps
		r_t = tf.transpose(r_t, perm=[1, 0, 2])
		r_t_sliced = tf.convert_to_tensor([r_t[t:t+n_step_q, :, :] for t in range(window_size)], dtype=tf.float32)
		r_t_sliced = tf.squeeze(tf.transpose(r_t_sliced, perm=[0, 2, 1, 3]))
		r_t_sl_shape = r_t_sliced.shape
		# reshape (batch, T, n) -> (batch*T, n)
		r_t_sliced = tf.reshape(r_t_sliced, [r_t_sl_shape[0]*r_t_sl_shape[1], r_t_sl_shape[2]])
		# r_t_slices [T*batch, n_steps] x  discounts [n_steps, 1]
		r_t = tf.linalg.matvec(r_t_sliced, discounts)
		r_t = tf.reshape(r_t, [r_t_sl_shape[0], r_t_sl_shape[1]])

	# reshape again to make tensors compatible with trfl API
	q_tm1_shape = q_tm1.shape
	q_tm1 = tf.reshape(q_tm1, [q_tm1_shape[0]*q_tm1_shape[1], q_tm1_shape[2]])
	q_t = tf.reshape(q_t, [q_tm1_shape[0]*q_tm1_shape[1], q_tm1_shape[2]])
	a_tm1_shape = a_tm1.shape
	a_tm1 = tf.squeeze(tf.reshape(a_tm1, [a_tm1_shape[0]*a_tm1_shape[1], a_tm1_shape[2]]))
	r_t_shape = r_t.shape
	r_t = tf.reshape(r_t, [r_t_shape[0] * r_t_shape[1]])
	dones_shape = dones.shape
	dones = tf.reshape(dones, [dones_shape[0]*dones_shape[1]])

	p_cont = 0.0
	if n_step_q == 1:
		# discount factor (at t=1) for bootstrapped value
		p_cont = tf.math.multiply(tf.ones((dones.shape)) - dones, gamma)
	elif n_step_q > 1:
		# discount factor (at t=n+1) accordingly for bootstrapped value
		p_cont = tf.math.multiply(tf.ones((dones.shape)) - dones, tf.math.pow(gamma, n_step_q))

	loss, extra = 0.0, None
	if not double_q:
		loss, extra = trfl.qlearning(q_tm1, a_tm1, r_t, p_cont, q_t)
	elif double_q:
		q_t_selector = tf.convert_to_tensor(q_t_selector_seq, dtype=tf.float32)
		q_t_selector = tf.reshape(q_t_selector, [q_tm1_shape[0] * q_tm1_shape[1], q_tm1_shape[2]])
		loss, extra = trfl.double_qlearning(q_tm1, a_tm1, r_t, p_cont, q_t, q_t_selector)

	# average over batch_dim = (batch*time)
	loss = tf.reduce_mean(loss, axis=0)
	# print("Inside q_learning bellman updates took %4.5f" % (time.time() - q_backup_start))
	return loss, extra
