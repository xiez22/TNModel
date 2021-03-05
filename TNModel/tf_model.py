import tensorflow as tf
import tensornetwork as tn
import TNModel.simple_mps as simple_mps

class MPSLayer(tf.keras.layers.Layer):
	def __init__(self, hyper_params):
		super(MPSLayer, self).__init__()

		self.hyper_params = hyper_params
		self.single_rank = hyper_params['rank']//2
		self.vectorized = hyper_params['vectorized']

		mps = simple_mps.simple_mps(
			nodes=hyper_params['rank']+1,
			bond_dim=hyper_params['bond_dim'],
			phys_dim=[hyper_params['phys_dim']]*self.single_rank+[hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
			std=1e-3,
			backend='tensorflow'
		)

		self.mps_var = [tf.Variable(node, name=f'mps{i}', trainable=True, dtype=tf.float32) for (i, node) in enumerate(mps)]


	def call(self, inputs):
		def f(input_vec, mps_var):
			input_vec = tf.reshape(input_vec, (28*28, 2))

			mps_nodes = []
			input_nodes = []
			for i in mps_var:
				mps_nodes.append(tn.Node(i))
			for i in range(input_vec.shape[0]):
				input_nodes.append(tn.Node(input_vec[i]))

			# Connect the edges
			mps_nodes[0][0]^mps_nodes[1][0]
			for i in range(1, 28*28):
				mps_nodes[i][2]^mps_nodes[i+1][0]
			for i in range(28*28//2):
				mps_nodes[i][1] ^ input_nodes[i][0]
			for i in range(28*28//2, 28*28):
				mps_nodes[i+1][1] ^ input_nodes[i][0]

			# Contract
			for i in range(28*28//2):
				mps_nodes[i] = mps_nodes[i] @ input_nodes[i]
			for i in range(28*28//2, 28*28):
				mps_nodes[i+1] = mps_nodes[i+1] @ input_nodes[i]

			left = mps_nodes[0]
			for i in range(1, 28*28//2):
				left = left @ mps_nodes[i]
			
			right = mps_nodes[28*28]
			for i in range(28*28-1, 28*28//2, -1):
				right = right @ mps_nodes[i]

			ans = tf.einsum('a,aba,a->b', left.tensor, mps_nodes[28*28//2].tensor, right.tensor)
			ans = tf.reshape(ans, [10])

			return ans

		result = None
		if self.vectorized:
			result = tf.vectorized_map(
				lambda vec: f(vec, self.mps_var),
				inputs
			)
		else:
			result = tf.map_fn(
				lambda vec: f(vec, self.mps_var),
				inputs
			)

		return tf.reshape(result, [-1, 10])


class SBS1dLayer(tf.keras.layers.Layer):
	def __init__(self, hyper_params):
		super(SBS1dLayer, self).__init__()

		self.hyper_params = hyper_params
		self.single_rank = hyper_params['rank']//2
		self.xnodes = hyper_params['rank']+1
		self.ynodes = hyper_params['string_cnt']
		self.vectorized = hyper_params['vectorized']
		self.sbs_op = hyper_params['sbs_op']

		mps = [simple_mps.simple_mps(
			nodes=self.xnodes,
			bond_dim=hyper_params['bond_dim'],
			phys_dim=[hyper_params['phys_dim']]*self.single_rank+[hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
			std=1e-3,
			backend='tensorflow'
		) for _ in range(self.ynodes)]

		self.mps_var = [[tf.Variable(node, name=f'mps{i}', trainable=True, dtype=tf.float32) for (i, node) in enumerate(j)] for j in mps]

	@staticmethod
	def func(inputs, mps, vectorized=False):
		def f(input_vec, mps_var):
			input_vec = tf.reshape(input_vec, (28*28, 2))

			mps_nodes = []
			input_nodes = []
			for i in mps_var:
				mps_nodes.append(tn.Node(i))
			for i in range(input_vec.shape[0]):
				input_nodes.append(tn.Node(input_vec[i]))

			# Connect the edges
			mps_nodes[0][0]^mps_nodes[1][0]
			for i in range(1, 28*28):
				mps_nodes[i][2]^mps_nodes[i+1][0]
			for i in range(28*28//2):
				mps_nodes[i][1] ^ input_nodes[i][0]
			for i in range(28*28//2, 28*28):
				mps_nodes[i+1][1] ^ input_nodes[i][0]

			# Contract
			for i in range(28*28//2):
				mps_nodes[i] = mps_nodes[i] @ input_nodes[i]
			for i in range(28*28//2, 28*28):
				mps_nodes[i+1] = mps_nodes[i+1] @ input_nodes[i]

			left = mps_nodes[0]
			for i in range(1, 28*28//2):
				left = left @ mps_nodes[i]
			
			right = mps_nodes[28*28]
			for i in range(28*28-1, 28*28//2, -1):
				right = right @ mps_nodes[i]

			ans = tf.einsum('a,aba,a->b', left.tensor, mps_nodes[28*28//2].tensor, right.tensor)
			ans = tf.reshape(ans, [10])

			return ans

		result = None
		if not vectorized:
			result = tf.map_fn(
				lambda vec: f(vec, mps),
				inputs
			)
		else:
			result = tf.vectorized_map(
				lambda vec: f(vec, mps),
				inputs
			)

		return tf.reshape(result, [-1, 10])

	def call(self, inputs):
		if self.sbs_op == 'prod':
			result = None
			for mps in self.mps_var:
				if result is None:
					result = self.func(inputs, mps, vectorized=self.vectorized)
				else:
					result = result * self.func(inputs, mps, vectorized=self.vectorized)
		elif self.sbs_op == 'mean':
			result = []
			for mps in self.mps_var:
				result.append(self.func(inputs, mps, vectorized=self.vectorized))
			result = tf.reduce_mean(tf.stack(result, axis=0), axis=0)
		else:
			raise NotImplementedError()

		return result
