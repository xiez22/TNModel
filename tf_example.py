import numpy as np
import tensorflow as tf
import tensornetwork as tn
import TNModel.tf_model as tf_model
from TNModel.utils import *

# If you want to run in eager mode, just comment those two lines.
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

tn.set_default_backend('tensorflow')

# HyperParams
hyper_params = {
	'rank': 28*28,
	'phys_dim': 2,
	'bond_dim': 3,
	'labels': 10,
	'string_cnt': 3,  # for 1d-sbs only
	'sbs_op': 'mean',  # mean or prod , alternative choice for 1d-sbs contraction
	'model': 'mps',  # mps (finished) or 1d-sbs (half-working)
	'vectorized': True # vectorized_map is only supported in part of the machines.
}


if __name__ == '__main__':
	# Datasets
	mnist = tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	train_cos = (1.0 - x_train).reshape(-1, 28*28, 1)
	train_sin = x_train.reshape(-1, 28*28, 1)

	test_cos = (1.0 - x_test).reshape(-1, 28*28, 1)
	test_sin = x_test.reshape(-1, 28*28, 1)

	x_train = np.concatenate((train_cos, train_sin), axis=2)
	x_test = np.concatenate((test_cos, test_sin), axis=2)

	# Model
	print('Building model...')

	if hyper_params['model'] == 'mps':
		model = tf.keras.models.Sequential([
			tf_model.MPSLayer(hyper_params=hyper_params),
			tf.keras.layers.Softmax()
		])
	elif hyper_params['model'] == '1d-sbs': 
		model = tf.keras.models.Sequential([
			tf_model.SBS1dLayer(hyper_params=hyper_params),
			tf.keras.layers.Softmax()
		])
	else:
		raise NotImplementedError()

	print('Compiling model...')
	model.compile(
		optimizer='rmsprop',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']	
	)
	print('Finished!')

	hist = LossHistory()
	model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=128, use_multiprocessing=True, callbacks=[hist])
	hist.loss_plot(loss_type='batch')

	result = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
	print('Evaluate Results:', result)
