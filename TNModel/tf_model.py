import tensorflow as tf
import tensornetwork as tn
import TNModel.simple_mps as simple_mps
import TNModel.simple_peps as simple_peps
from typing import List, Tuple


class MPSLayer(tf.keras.layers.Layer):
    def __init__(self, hyper_params):
        super(MPSLayer, self).__init__()

        self.hyper_params = hyper_params
        self.single_rank = hyper_params['rank']//2
        self.vectorized = hyper_params['vectorized']

        mps = simple_mps.simple_mps(
            nodes=hyper_params['rank']+1,
            bond_dim=hyper_params['bond_dim'],
            phys_dim=[hyper_params['phys_dim']]*self.single_rank +
            [hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
            std=1e-3,
            backend='tensorflow'
        )

        self.mps_var = [tf.Variable(node, name=f'mps{i}', trainable=True, dtype=tf.float32) for (
            i, node) in enumerate(mps)]

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
            mps_nodes[0][0] ^ mps_nodes[1][0]
            for i in range(1, 28*28):
                mps_nodes[i][2] ^ mps_nodes[i+1][0]
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

            ans = tf.einsum('a,aba,a->b', left.tensor,
                            mps_nodes[28*28//2].tensor, right.tensor)
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
            phys_dim=[hyper_params['phys_dim']]*self.single_rank +
            [hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
            std=1e-3,
            backend='tensorflow'
        ) for _ in range(self.ynodes)]

        self.mps_var = [[tf.Variable(node, name=f'mps{i}', trainable=True, dtype=tf.float32) for (
            i, node) in enumerate(j)] for j in mps]

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
            mps_nodes[0][0] ^ mps_nodes[1][0]
            for i in range(1, 28*28):
                mps_nodes[i][2] ^ mps_nodes[i+1][0]
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

            ans = tf.einsum('a,aba,a->b', left.tensor,
                            mps_nodes[28*28//2].tensor, right.tensor)
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
                    result = result * \
                        self.func(inputs, mps, vectorized=self.vectorized)
        elif self.sbs_op == 'mean':
            result = []
            for mps in self.mps_var:
                result.append(
                    self.func(inputs, mps, vectorized=self.vectorized))
            result = tf.reduce_mean(tf.stack(result, axis=0), axis=0)
        else:
            raise NotImplementedError()

        return result


class PEPSLayer(tf.keras.layers.Layer):
    def __init__(self, features_in, features_out, xnodes, ynodes, max_singular_values=16, bond_dim=6, device='cpu', vectorized=False):
        super(PEPSLayer, self).__init__()

        self.features_in = features_in
        self.features_out = features_out
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.device = device
        self.max_singular_values = max_singular_values
        self.vectorized = vectorized

        peps, self.index_result = simple_peps.simple_peps(
            xnodes=xnodes,
            ynodes=ynodes,
            bond_dim=bond_dim,
            features_in=features_in,
            features_out=features_out,
            std=1e-2,
            init_method='uniform'
        )

        self.peps_var = [[tf.Variable(i, name=f'peps_{index0}_{index1}', dtype=tf.float32, trainable=True)
                          for index1, i in enumerate(line)] for index0, line in enumerate(peps)]

    def call(self, inputs):
        def func(inputs, xnodes, ynodes, peps_var, index_result):
            # C * x_nodes * y_nodes
            peps_nodes = []
            input_nodes = []

            for i in range(xnodes):
                peps_line = []
                input_line = []
                for j in range(ynodes):
                    peps_line.append(
                        tn.Node(peps_var[i][j], name=f'p_{i}_{j}'))
                    input_line.append(
                        tn.Node(inputs[i, j, :], name=f'i_{i}_{j}'))
                peps_nodes.append(peps_line)
                input_nodes.append(input_line)

            # Connect the edges
            cx, cy = xnodes//2, ynodes//2

            # Input Features
            for i in range(xnodes):
                for j in range(ynodes):
                    input_nodes[i][j][0] ^ peps_nodes[i][j][0]

            # Y Bond
            for i in range(xnodes):
                for j in range(ynodes-1):
                    index1 = index_result[i, j, 1]
                    index2 = index_result[i, j+1, 3]
                    peps_nodes[i][j][index1] ^ peps_nodes[i][j+1][index2]

            # X Bond
            for j in range(ynodes):
                for i in range(xnodes-1):
                    index1 = index_result[i, j, 0]
                    index2 = index_result[i+1, j, 2]
                    peps_nodes[i][j][index1] ^ peps_nodes[i+1][j][index2]

            # Contract
            # Contract the features
            for i in range(xnodes):
                for j in range(ynodes):
                    input_nodes[i][j] = input_nodes[i][j] @ peps_nodes[i][j]
                    input_nodes[i][j].name = f'p_{i}_{j}'
                    input_nodes[i][j].tensor = input_nodes[i][j].tensor / \
                        tf.norm(input_nodes[i][j].tensor)

            # Contract each row
            left_nodes: List[tn.Node] = input_nodes[0]
            right_nodes: List[tn.Node] = input_nodes[xnodes-1]
            middle_nodes: List[tn.Node] = input_nodes[cx]

            for i in range(1, cx):
                for j in range(ynodes):
                    left_nodes[j] = left_nodes[j] @ input_nodes[i][j]
                    left_nodes[j].name = f'l_{j}'

                # Row Normalization
                row_norm = tf.reduce_mean(tf.stack(
                    [tf.norm(t.tensor) for t in left_nodes]))
                for t in left_nodes:
                    t.tensor = t.tensor / row_norm

            for i in range(xnodes-2, cx, -1):
                for j in range(ynodes):
                    right_nodes[j] = right_nodes[j]@input_nodes[i][j]
                    right_nodes[j].name = f'r_{j}'

                # Row Normalization
                row_norm = tf.reduce_mean(tf.stack(
                    [tf.norm(t.tensor) for t in right_nodes]))
                for t in right_nodes:
                    t.tensor = t.tensor / row_norm

            for j in range(ynodes):
                middle_nodes[j] = left_nodes[j] @ middle_nodes[j]

            for j in range(ynodes):
                middle_nodes[j] = right_nodes[j] @ middle_nodes[j]

            down_node = middle_nodes[0]
            up_node = middle_nodes[ynodes-1]

            for j in range(1, cy+1):
                down_node = down_node @ middle_nodes[j]
                down_node.tensor = down_node.tensor / tf.norm(down_node.tensor)

            for j in range(ynodes-2, cy, -1):
                up_node = up_node @ middle_nodes[j]
                up_node.tensor = up_node.tensor / tf.norm(up_node.tensor)

            result = (down_node @ up_node).tensor

            result = tf.reshape(result, [10]) / tf.norm(result)
            return result

        result = None

        if self.vectorized:
            result = tf.vectorized_map(
                lambda vec: func(
                    vec, self.xnodes, self.ynodes, self.peps_var, self.index_result),
                inputs
            )
        else:
            result = tf.map_fn(
                lambda vec: func(vec, self.xnodes, self.ynodes,
                                 self.peps_var, self.index_result),
                inputs
            )

        result = tf.reshape(result, [-1, 10])
        return result


class PEPSCNNLayer(tf.keras.layers.Layer):
    def __init__(self, hyper_params):
        super(PEPSCNNLayer, self).__init__()

        self.hyper_params = hyper_params

        self.cnn_layer = tf.keras.layers.Conv2D(
            filters=hyper_params['phys_dim'],
            kernel_size=(4, 4),
            strides=(4, 4)
        )

        self.peps_layer = PEPSLayer(
            features_in=hyper_params['phys_dim'],
            features_out=10,
            xnodes=28//4,
            ynodes=28//4,
            bond_dim=hyper_params['bond_dim'],
            max_singular_values=hyper_params['max_singular_values'],
            vectorized=hyper_params['vectorized']
        )

    def call(self, inputs):
        inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])

        out = tf.sigmoid(self.cnn_layer(inputs))
        out = out / tf.abs(tf.reduce_sum(out, axis=1, keepdims=True))

        out = self.peps_layer(out)

        return out
