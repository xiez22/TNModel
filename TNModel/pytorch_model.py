import torch
import torch.nn as nn
import tensornetwork as tn
import TNModel.simple_mps as simple_mps
import TNModel.simple_peps as simple_peps
from tensornetwork import contractors
from typing import List, Tuple


class MPSLayer(nn.Module):
    def __init__(self, hyper_params):
        super(MPSLayer, self).__init__()

        self.hyper_params = hyper_params
        self.single_rank = hyper_params['rank']//2

        mps = simple_mps.simple_mps(
            nodes=hyper_params['rank']+1,
            bond_dim=hyper_params['bond_dim'],
            phys_dim=[hyper_params['phys_dim']]*self.single_rank +
            [hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
            std=1e-3
        )

        self.mps_var = [nn.Parameter(i, requires_grad=True) for i in mps]

        for i, v in enumerate(self.mps_var):
            self.register_parameter(f'mps{i}', param=v)

    def forward(self, inputs):
        def f(input_vec, mps_var):
            input_vec = torch.reshape(input_vec, (28*28, 2))

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

            ans = mps_nodes[0]
            for i in range(1, 28*28+1):
                ans = ans @ mps_nodes[i]

            # ans = contractors.auto(mps_nodes+input_nodes).tensor
            ans = torch.reshape(ans.tensor, [10])

            return ans

        ans = torch.stack([f(vec, self.mps_var) for vec in inputs], dim=0)
        return ans


class SBS1dLayer(nn.Module):
    def __init__(self, hyper_params):
        super(SBS1dLayer, self).__init__()

        self.hyper_params = hyper_params
        self.single_rank = hyper_params['rank']//2
        self.xnodes = hyper_params['rank']+1
        self.ynodes = hyper_params['string_cnt']

        mps = [simple_mps.simple_mps(
            nodes=self.xnodes,
            bond_dim=hyper_params['bond_dim'],
            phys_dim=[hyper_params['phys_dim']]*self.single_rank +
            [hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
            std=1e-3
        ) for _ in range(self.ynodes)]

        self.mps_var = [[nn.Parameter(i, requires_grad=True)
                         for i in j] for j in mps]

        for i, v in enumerate(self.mps_var):
            for j, s in enumerate(v):
                self.register_parameter(f'mps{i}_{j}', param=s)

    @staticmethod
    def func(inputs, mps):
        def f(input_vec, mps_var):
            input_vec = torch.reshape(input_vec, (28*28, 2))

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

            ans = mps_nodes[0]
            for i in range(1, 28*28+1):
                ans = ans @ mps_nodes[i]

            # ans = contractors.auto(mps_nodes+input_nodes).tensor
            ans = torch.reshape(ans.tensor, [10])

            return ans

        ans = torch.stack([f(vec, mps) for vec in inputs], dim=0)
        return ans

    def forward(self, inputs):
        ans = torch.ones(inputs.shape[0], 10).float()

        for i in range(self.ynodes):
            ans = ans * self.func(inputs, self.mps_var[i])

        return ans


class PEPSLayer(nn.Module):
    def __init__(self, features_in, features_out, xnodes, ynodes, max_singular_values=16, bond_dim=6, device='cpu'):
        super(PEPSLayer, self).__init__()

        self.features_in = features_in
        self.features_out = features_out
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.device = device
        self.max_singular_values = max_singular_values

        peps, self.index_result = simple_peps.simple_peps(
            xnodes=xnodes,
            ynodes=ynodes,
            bond_dim=bond_dim,
            features_in=features_in,
            features_out=features_out,
            std=1e-2,
            init_method='uniform'
        )

        self.peps_var = [[nn.Parameter(torch.from_numpy(
            i).float(), requires_grad=True) for i in line] for line in peps]

        for i, line in enumerate(self.peps_var):
            for j, node in enumerate(line):
                self.register_parameter(f'peps_{i}_{j}', node)

    def func(self, inputs):
        # C * x_nodes * y_nodes
        peps_nodes = []
        input_nodes = []

        for i in range(self.xnodes):
            peps_line = []
            input_line = []
            for j in range(self.ynodes):
                peps_line.append(
                    tn.Node(self.peps_var[i][j], name=f'p_{i}_{j}'))
                input_line.append(tn.Node(inputs[:, i, j], name=f'i_{i}_{j}'))
            peps_nodes.append(peps_line)
            input_nodes.append(input_line)

        # Connect the edges
        cx, cy = self.xnodes//2, self.ynodes//2

        # Input Features
        for i in range(self.xnodes):
            for j in range(self.ynodes):
                input_nodes[i][j][0] ^ peps_nodes[i][j][0]

        # Y Bond
        for i in range(self.xnodes):
            for j in range(self.ynodes-1):
                index1 = self.index_result[i, j, 1]
                index2 = self.index_result[i, j+1, 3]
                peps_nodes[i][j][index1] ^ peps_nodes[i][j+1][index2]

        # X Bond
        for j in range(self.ynodes):
            for i in range(self.xnodes-1):
                index1 = self.index_result[i, j, 0]
                index2 = self.index_result[i+1, j, 2]
                peps_nodes[i][j][index1] ^ peps_nodes[i+1][j][index2]

        # Contract
        # Contract the features
        contracted_nodes = []
        for i in range(self.xnodes):
            for j in range(self.ynodes):
                input_nodes[i][j] = input_nodes[i][j] @ peps_nodes[i][j]
                input_nodes[i][j].name = f'p_{i}_{j}'
                input_nodes[i][j].tensor = input_nodes[i][j].tensor / \
                    input_nodes[i][j].tensor.norm()

                contracted_nodes.append(input_nodes[i][j])

        # # Contract each row
        # left_nodes: List[tn.Node] = input_nodes[0]
        # right_nodes: List[tn.Node] = input_nodes[self.xnodes-1]
        # middle_nodes: List[tn.Node] = input_nodes[cx]

        # for i in range(1, cx):
        #     for j in range(self.ynodes):
        #         left_nodes[j] = left_nodes[j] @ input_nodes[i][j]
        #         left_nodes[j].name = f'l_{j}'

        #     # # Row Normalization
        #     # row_norm = torch.mean(torch.stack(
        #     #     [t.tensor.norm() for t in left_nodes]))
        #     # for t in left_nodes:
        #     #     t.tensor = t.tensor / row_norm

        #     # # RQ Decomposition
        #     # for j in range(self.ynodes-1):
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in left_nodes[j].edges:
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'l_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name[0] == 'p':
        #     #             right_edges.append(edge)
        #     #         elif nxt_node_name == f'l_{j-1}':
        #     #             right_edges.append(edge)
        #     #         else:
        #     #             left_edges.append(edge)

        #     #     node1, node2 = tn.split_node_rq(
        #     #         left_nodes[j], left_edges=left_edges, right_edges=right_edges)
        #     #     left_nodes[j] = node2
        #     #     left_nodes[j+1] = left_nodes[j+1] @ node1
        #     #     # left_nodes[j+1].tensor = left_nodes[j+1].tensor / \
        #     #     #     left_nodes[j+1].tensor.norm()
        #     #     left_nodes[j].name = f'l_{j}'
        #     #     left_nodes[j+1].name = f'l_{j+1}'

        #     # # SVD
        #     # for j in range(self.ynodes-1, 0, -1):
        #     #     tmp_node = left_nodes[j] @ left_nodes[j-1]
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in tmp_node.edges:
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'l_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name == f'p_{i+1}_{j}':
        #     #             left_edges.append(edge)
        #     #         elif nxt_node_name == f'p_{i+1}_{j-1}':
        #     #             right_edges.append(edge)
        #     #         elif nxt_node_name == f'l_{j+1}':
        #     #             left_edges.append(edge)
        #     #         else:
        #     #             right_edges.append(edge)

        #     #     node1, node2, _ = tn.split_node(
        #     #         tmp_node, left_edges=left_edges, right_edges=right_edges, max_singular_values=self.max_singular_values)

        #     #     left_nodes[j] = node1
        #     #     left_nodes[j-1] = node2
        #     #     left_nodes[j].name = f'l_{j}'
        #     #     left_nodes[j-1].name = f'l_{j-1}'

        #     #     # QR Decomposition
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in left_nodes[j].edges:
        #     #         if not edge.node2 and not edge.node1:
        #     #             continue
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'l_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name[0] == 'p':
        #     #             left_edges.append(edge)
        #     #         elif nxt_node_name == f'l_{j+1}':
        #     #             left_edges.append(edge)
        #     #         else:
        #     #             right_edges.append(edge)

        #     #     node1, node2 = tn.split_node_qr(
        #     #         left_nodes[j], left_edges=left_edges, right_edges=right_edges)

        #     #     left_nodes[j] = node1
        #     #     left_nodes[j].name = f'l_{j}'
        #     #     left_nodes[j-1] = node2 @ left_nodes[j-1]
        #     #     # left_nodes[j-1].tensor = left_nodes[j-1].tensor / \
        #     #     #     left_nodes[j-1].tensor.norm()
        #     #     left_nodes[j-1].name = f'l_{j-1}'

        # for i in range(self.xnodes-2, cx, -1):
        #     for j in range(self.ynodes):
        #         right_nodes[j] = right_nodes[j]@input_nodes[i][j]
        #         right_nodes[j].name = f'r_{j}'

        #     # # Row Normalization
        #     # row_norm = torch.mean(torch.stack(
        #     #     [t.tensor.norm() for t in right_nodes]))
        #     # for t in right_nodes:
        #     #     t.tensor = t.tensor / row_norm

        #     # # RQ Decomposition
        #     # for j in range(self.ynodes-1):
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in right_nodes[j].edges:
        #     #         if not edge.node2 and not edge.node1:
        #     #             continue
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'r_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name[0] == 'p':
        #     #             right_edges.append(edge)
        #     #         elif nxt_node_name == f'r_{j-1}':
        #     #             right_edges.append(edge)
        #     #         else:
        #     #             left_edges.append(edge)

        #     #     node1, node2 = tn.split_node_rq(
        #     #         right_nodes[j], left_edges=left_edges, right_edges=right_edges)
        #     #     right_nodes[j] = node2
        #     #     right_nodes[j+1] = right_nodes[j+1] @ node1
        #     #     # right_nodes[j+1].tensor = right_nodes[j+1].tensor / \
        #     #     #     right_nodes[j+1].tensor.norm()
        #     #     right_nodes[j].name = f'r_{j}'
        #     #     right_nodes[j+1].name = f'r_{j+1}'

        #     # # SVD
        #     # for j in range(self.ynodes-1, 0, -1):
        #     #     tmp_node = right_nodes[j] @ right_nodes[j-1]
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in tmp_node.edges:
        #     #         if not edge.node2 and not edge.node1:
        #     #             continue
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'r_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name == f'p_{i-1}_{j}':
        #     #             left_edges.append(edge)
        #     #         elif nxt_node_name == f'p_{i-1}_{j-1}':
        #     #             right_edges.append(edge)
        #     #         elif nxt_node_name == f'r_{j+1}':
        #     #             left_edges.append(edge)
        #     #         else:
        #     #             right_edges.append(edge)

        #     #     node1, node2, _ = tn.split_node(
        #     #         tmp_node, left_edges=left_edges, right_edges=right_edges, max_singular_values=self.max_singular_values)

        #     #     right_nodes[j] = node1
        #     #     right_nodes[j-1] = node2
        #     #     right_nodes[j].name = f'r_{j}'
        #     #     right_nodes[j-1].name = f'r_{j-1}'

        #     #     # QR Decomposition
        #     #     left_edges = []
        #     #     right_edges = []

        #     #     for edge in right_nodes[j].edges:
        #     #         if not edge.node2 and not edge.node1:
        #     #             continue
        #     #         nxt_node_name = edge.node1.name if edge.node1.name != f'r_{j}' and edge.node1.name != '__unnamed_node__' else edge.node2.name

        #     #         if nxt_node_name[0] == 'p':
        #     #             left_edges.append(edge)
        #     #         elif nxt_node_name == f'r_{j+1}':
        #     #             left_edges.append(edge)
        #     #         else:
        #     #             right_edges.append(edge)

        #     #     node1, node2 = tn.split_node_qr(
        #     #         right_nodes[j], left_edges=left_edges, right_edges=right_edges)

        #     #     right_nodes[j] = node1
        #     #     right_nodes[j].name = f'r_{j}'
        #     #     right_nodes[j-1] = node2 @ right_nodes[j-1]
        #     #     # right_nodes[j-1].tensor = right_nodes[j-1].tensor / \
        #     #     #     right_nodes[j-1].tensor.norm()
        #     #     right_nodes[j-1].name = f'r_{j-1}'

        # for j in range(self.ynodes):
        #     middle_nodes[j] = left_nodes[j] @ middle_nodes[j]
        #     # middle_nodes[j].tensor = middle_nodes[j].tensor / \
        #     #     middle_nodes[j].tensor.norm()

        # for j in range(self.ynodes):
        #     middle_nodes[j] = right_nodes[j] @ middle_nodes[j]
        #     # middle_nodes[j].tensor = middle_nodes[j].tensor / \
        #     #     middle_nodes[j].tensor.norm()

        # down_node = middle_nodes[0]
        # up_node = middle_nodes[self.ynodes-1]

        # for j in range(1, cy+1):
        #     down_node = down_node @ middle_nodes[j]
        #     # down_node.tensor = down_node.tensor / down_node.tensor.norm()

        # for j in range(self.ynodes-2, cy, -1):
        #     up_node = up_node @ middle_nodes[j]
        #     # up_node.tensor = up_node.tensor / up_node.tensor.norm()

        # result = (down_node @ up_node).tensor

        # Contract the remaining peps (With Auto Mode)
        result = contractors.auto(contracted_nodes).tensor
        # print(result[0].item())

        result = result.view([10]) / result.norm()
        return result

    def forward(self, inputs):
        result = []
        for batch in inputs:
            result.append(self.func(batch))

        result = torch.stack(result, dim=0)
        return result


class PEPSCNNLayer(nn.Module):
    def __init__(self, hyper_params):
        super(PEPSCNNLayer, self).__init__()

        self.hyper_params = hyper_params
        self.cnn_layer = nn.Conv2d(
            in_channels=1,
            out_channels=hyper_params['phys_dim'],
            kernel_size=(4, 4),
            stride=(4, 4)
        )

        self.peps_layer = PEPSLayer(
            features_in=hyper_params['phys_dim'],
            features_out=10,
            xnodes=28//4,
            ynodes=28//4,
            bond_dim=hyper_params['bond_dim'],
            max_singular_values=hyper_params['max_singular_values']
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, 28, 28)

        out = torch.sigmoid(self.cnn_layer(inputs))
        out = out / torch.abs(torch.sum(out, dim=1, keepdim=True))

        out = self.peps_layer(out)

        return out
