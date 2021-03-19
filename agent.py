import tensorflow as tf
import sonnet as snt
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import _base
from baselines.common.schedules import LinearSchedule
import rl_loss

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE


class RecurrentQNet(tf.keras.Model):
	def __init__(self, num_units, n_actions, batch_size):
		super(RecurrentQNet, self).__init__()
		self.lstm = tf.keras.layers.LSTMCell(units=num_units)
		self.q_value_head = tf.keras.layers.Dense(units=n_actions, activation="linear")
		self.num_actions = n_actions
		self.batch_size = batch_size
		self.eps = tf.Variable(0., trainable=False)
		self.zero_state = self.lstm.get_initial_state(batch_size=self.batch_size,
													  dtype=tf.float32)

	def call(self, inputs, init_state, training=True):
		o_t, [h_t, c_t] = self.lstm(inputs, states=init_state, training=training)
		action_value = self.q_value_head(o_t)
		return action_value, h_t, c_t

	def step(self, inputs, init_state, update_eps, training=True, stochastic=True):
		q_values, h_t, c_t = self.call(inputs, init_state, training=training)
		greedy_actions = tf.math.argmax(q_values, axis=1)
		batch_size = tf.shape(inputs)[0]
		random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
		chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0.0, maxval=1.0, dtype=tf.float32) > update_eps
		stochastic_actions = tf.where(chose_random, greedy_actions, random_actions)

		if stochastic:
			output_actions = stochastic_actions
		elif not stochastic:
			output_actions = greedy_actions

		if not update_eps >= 0:
			self.eps.assign(update_eps)

		return output_actions, h_t, c_t


def exploration_policy(num_iters, exp_fraction, final_eps):
	# eps-greedy exploration
	return LinearSchedule(schedule_timesteps=int(exp_fraction*num_iters), initial_p=1.0, final_p=final_eps)


class KptConvEncoder(tf.keras.Model):
	def __init__(self, feature_dim, agent_size):
		super(KptConvEncoder, self).__init__()
		self.conv_1 = tf.keras.layers.Conv2D(filters=feature_dim, kernel_size=3, strides=1, padding="same")
		self.bn_1 = tf.keras.layers.BatchNormalization()
		self.relu_1 = tf.keras.layers.ReLU()

		self.conv_2 = tf.keras.layers.Conv2D(filters=feature_dim, kernel_size=3, strides=1, padding="same")
		self.bn_2 = tf.keras.layers.BatchNormalization()
		self.relu_2 = tf.keras.layers.ReLU()

		self.conv_3 = tf.keras.layers.Conv2D(filters=feature_dim, kernel_size=3, strides=2, padding="same")
		self.bn_3 = tf.keras.layers.BatchNormalization()
		self.relu_3 = tf.keras.layers.ReLU()

		self.conv_4 = tf.keras.layers.Conv2D(filters=feature_dim, kernel_size=3, strides=1, padding="same")
		self.bn_4 = tf.keras.layers.BatchNormalization()
		self.relu_4 = tf.keras.layers.ReLU()

		self.flatten = tf.keras.layers.Flatten()
		self.dense_1 = tf.keras.layers.Dense(units=agent_size)
		self.relu_d = tf.keras.layers.ReLU()

		# useful vars
		self.filters = [feature_dim, feature_dim, feature_dim, feature_dim]
		self.kernels = [3, 3, 3, 3]
		self.strides = [1, 1, 2, 1]

	def call(self, inputs, training=True):
		h1 = self.relu_1(self.bn_1(self.conv_1(inputs), training=training))
		h2 = self.relu_2(self.bn_2(self.conv_2(h1), training=training))
		h3 = self.relu_3(self.bn_3(self.conv_3(h2), training=training))
		h4 = self.relu_4(self.bn_4(self.conv_4(h3), training=training))
		flatten_h4 = self.flatten(h4)
		h_dense = self.relu_d(self.dense_1(flatten_h4))
		return h_dense


class PositionalEncoder(tf.keras.Model):
	def __init__(self, d_model):
		super(PositionalEncoder, self).__init__()
		self.d_model = d_model

		self.dense_1 = tf.keras.layers.Dense(units=64)
		self.bn_1 = tf.keras.layers.BatchNormalization()
		self.relu_1 = tf.keras.layers.ReLU()

		self.w_p = tf.keras.layers.Dense(units=d_model, activation="linear")

	def call(self, inputs, training):
		h1 = self.relu_1(self.bn_1(self.dense_1(inputs), training=training))
		h2 = self.w_p(h1)
		return h2


class NodeEncoder(tf.keras.Model):
	def __init__(self, output_dim):
		super(NodeEncoder, self).__init__()
		self.dense_1 = tf.keras.layers.Dense(units=output_dim)
		self.bn_1 = tf.keras.layers.BatchNormalization()
		self.relu_1 = tf.keras.layers.ReLU()

		self.dense_2 = tf.keras.layers.Dense(units=output_dim)
		self.bn_2 = tf.keras.layers.BatchNormalization()
		self.relu_2 = tf.keras.layers.ReLU()

	def call(self, inputs, training):
		h1 = self.relu_1(self.bn_1(self.dense_1(inputs), training=training))
		h2 = self.relu_2(self.bn_2(self.dense_2(h1), training=training))
		return h2


class MLP(_base.AbstractModule):
	def __init__(self, hidden_size):
		super(MLP, self).__init__()
		self.dense_1 = snt.Linear(hidden_size, name="hidden1")
		self.bn_1 = snt.BatchNorm(create_scale=True, create_offset=True)

		self.dense_2 = snt.Linear(hidden_size, name="hidden2")
		self.bn_2 = snt.BatchNorm(create_scale=True, create_offset=True)

	def _build(self, inputs, **kwargs):
		h1 = tf.nn.relu(self.bn_1(self.dense_1(inputs), is_training=kwargs["is_training"]))
		output = tf.nn.relu(self.bn_2(self.dense_2(h1), is_training=kwargs["is_training"]))
		return output


class MLPGraphIndependent(_base.AbstractModule):
	"""GraphIndependent with separate MLP edge, node, and global models."""
	def __init__(self, latent_size, num_layers, name="MLPGraphIndependent"):
		super(MLPGraphIndependent, self).__init__(name=name)
		with self._enter_variable_scope():
			self._latent_size = latent_size
			self._num_layers = num_layers
			self._network = modules.GraphIndependent(edge_model_fn=lambda: MLP(latent_size),
													node_model_fn=lambda: MLP(latent_size))

	def _build(self, inputs, is_training):
		return self._network(inputs, edge_model_kwargs={"is_training": is_training},
							 node_model_kwargs={"is_training": is_training})


class KptGnnEncoder(_base.AbstractModule):
	"""A Graph Network model design  (https://arxiv.org/abs/1806.01261)
	using a "core" module Interaction Network and MLPGraphIndependent networks
	as Encoder and Decoder modules.
	The "core" Interaction Network, which performs N rounds of processing
	(message-passing) steps. Edges and Nodes are encoded by and decoded back
	from the "core" independently using a MLPGraphIndependent network.
	"""

	def __init__(self,
				latent_size=32,
				num_layers=2,
				decoder_size=64,
				name="kpt_gnn_enc"):
		super(KptGnnEncoder, self).__init__(name=name)
		self._latent_size = latent_size
		self._num_layers = num_layers
		self._is_training = None
		self._encoder = MLPGraphIndependent(latent_size, num_layers)
		self._interaction_core = modules.InteractionNetwork(
										lambda: MLP(self._latent_size),
										lambda: MLP(self._latent_size))
		self._decoder = MLPGraphIndependent(decoder_size, num_layers)

	def _build(self, v, num_processing_steps, is_training):
		# simply use kpts as nodes in the graph (no top-down attn)
		input_graphs = rl_loss.get_graph_tuple(v)
		# pre-process graphs-tuple data
		input_graphs = utils_tf.fully_connect_graph_static(
													input_graphs,
													exclude_self_edges=False)
		input_graphs = utils_tf.set_zero_edge_features(input_graphs,
								edge_size=self._latent_size)
		# encode input graphs
		latent = self._encoder(input_graphs, is_training)
		delta_latent = latent
		# measure interaction-effects with keypoints as nodes
		for _ in range(num_processing_steps):
			delta_latent = self._interaction_core(latent,
								edge_model_kwargs={"is_training": is_training},
								node_model_kwargs={"is_training": is_training})
		# decode last round latent graph
		output = self._decoder(delta_latent, is_training)
		return output
