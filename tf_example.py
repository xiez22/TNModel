from tensorflow.python.framework.ops import disable_eager_execution
from TNModel.utils import *
import TNModel.tf_model as tf_model
import tensornetwork as tn
import tensorflow as tf
import numpy as np
import os
os.environ['TF_DISABLE_MLC'] = '1'


# If you want to run in eager mode, just comment those two lines.
disable_eager_execution()

tn.set_default_backend('tensorflow')

# HyperParams
hyper_params = {
    'rank': 28*28,
    'phys_dim': 10,
    'bond_dim': 2,
    'labels': 10,
    'string_cnt': 4,  # for 1d-sbs only
    'sbs_op': 'mean',  # mean or prod , alternative choice for 1d-sbs contraction
    'model': 'peps',  # mps (finished) or 1d-sbs (half-working)
    # vectorized_map is only supported in part of the machines.
    'vectorized': True,
    'batch_size': 16,
    'max_singular_values': 64
}


if __name__ == '__main__':
    # Datasets
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if not hyper_params['model'] == 'peps':
        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_cos = (1.0 - x_train).reshape(-1, 28*28, 1)
        train_sin = x_train.reshape(-1, 28*28, 1)

        test_cos = (1.0 - x_test).reshape(-1, 28*28, 1)
        test_sin = x_test.reshape(-1, 28*28, 1)

        x_train = np.concatenate((train_cos, train_sin), axis=2)
        x_test = np.concatenate((test_cos, test_sin), axis=2)
    else:
        train_mean, train_std = np.mean(x_train), np.std(x_train)
        test_mean, test_std = np.mean(x_test), np.std(x_test)

        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - test_mean) / test_std

        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])

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
    elif hyper_params['model'] == 'peps':
        model = tf.keras.models.Sequential([
            tf_model.PEPSCNNLayer(hyper_params=hyper_params),
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

    if hyper_params['model'] == 'peps':
        print('Building model...')
        model.build((None, 1, 28, 28))
        model.summary()

    print('Finished!')

    hist = LossHistory()
    print('Start training...')
    model.fit(x_train, y_train, epochs=5, verbose=1,
              batch_size=hyper_params['batch_size'], callbacks=[hist])
    hist.loss_plot(loss_type='batch')

    result = model.evaluate(
        x_test, y_test, batch_size=hyper_params['batch_size'], verbose=1)
    print('Evaluate Results:', result)
