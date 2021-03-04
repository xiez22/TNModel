import numpy as np
import tensorflow as tf
import tensornetwork as tn
import TNModel.tf_model as tf_model

# If you want to run in eager mode, just comment those two lines.
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

tn.set_default_backend('tensorflow')

# HyperParams
hyper_params = {
	'rank': 28*28,
	'phys_dim': 2,
	'bond_dim': 6,
	'labels': 10,
	'string_cnt': 2,
	'model': 'mps',  # mps (finished) or 1d-sbs (runs with bugs)
	'vectorized': False  # vectorized_fn is only supported in part of the machines.
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
			tf_model.MPSLayer(hyper_params=hyper_params, vectorized=hyper_params['vectorized']),
			tf.keras.layers.Softmax()
		])
	elif hyper_params['model'] == '1d-sbs': 
		model = tf.keras.models.Sequential([
			tf_model.SBS1dLayer(hyper_params=hyper_params, vectorized=hyper_params['vectorized']),
			tf.keras.layers.Softmax()
		])
	else:
		raise NotImplementedError()

	print('Compiling model...')
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']	
	)
	print('Finished!')

	model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=8)
	model.evaluate(x_test, y_test, batch_size=1, verbose=1)
