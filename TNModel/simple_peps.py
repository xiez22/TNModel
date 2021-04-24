import numpy as np
import numpy.linalg as linalg


def simple_peps_tensor(shape, std=1e-2, init_method='uniform'):
    if init_method == 'uniform':
        a = np.random.uniform(0.0, 2.0, size=shape)
        a = a / linalg.norm(a)
        return a
    if init_method == 'eye':
        assert len(shape) >= 2 and shape[-2] == shape[-1]
        base = np.eye(shape[-1], dtype=np.float32)
        for i in range(-3, -len(shape)-1, -1):
            base = np.stack([base]*int(shape[i]), axis=0)
        return base + np.random.normal(size=base.shape) * std
    else:
        raise NotImplementedError()


def simple_peps(xnodes, ynodes, bond_dim, features_in, features_out, std=1e-2, init_method='uniform'):
    result = []
    index_result = []
    dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
    assert xnodes >= 2 and ynodes >= 2
    cx, cy = xnodes//2, ynodes//2

    for i in range(xnodes):
        line_index_result = []
        line = []
        for j in range(ynodes):
            shape = [features_in]
            cur_index_result = [None] * 4

            if i == cx and j == cy:
                shape.append(features_out)

            for k in range(4):
                nx, ny = i+dx[k], j+dy[k]
                if nx >= 0 and nx < xnodes and ny >= 0 and ny < ynodes:
                    cur_index_result[k] = len(shape)
                    shape.append(bond_dim)

            line.append(simple_peps_tensor(shape, std, init_method))
            line_index_result.append(cur_index_result)

        result.append(line)
        index_result.append(line_index_result)

    return result, np.array(index_result)
