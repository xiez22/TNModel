import torch
import torch.nn as nn
import tensornetwork as tn
import TNModel.simple_mps as simple_mps
import TNModel.simple_peps as simple_peps
from tensornetwork import contractors


class MPSLayer(nn.Module):
	def __init__(self, hyper_params):
		super(MPSLayer, self).__init__()

		self.hyper_params = hyper_params
		self.single_rank = hyper_params['rank']//2
		

		mps = simple_mps.simple_mps(
			nodes=hyper_params['rank']+1,
			bond_dim=hyper_params['bond_dim'],
			phys_dim=[hyper_params['phys_dim']]*self.single_rank+[hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
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
			phys_dim=[hyper_params['phys_dim']]*self.single_rank+[hyper_params['labels']]+[hyper_params['phys_dim']]*self.single_rank,
			std=1e-3
		) for _ in range(self.ynodes)]

		self.mps_var = [[nn.Parameter(i, requires_grad=True) for i in j] for j in mps]
		
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
	def __init__(self, features_in, features_out, xnodes, ynodes, bond_dim=6, device='cpu'):
		super(PEPSLayer, self).__init__()

		self.features_in = features_in
		self.features_out = features_out
		self.xnodes = xnodes
		self.ynodes = ynodes
		self.device = device

		peps, self.index_result = simple_peps.simple_peps(
			xnodes=xnodes, 
			ynodes=ynodes,
			bond_dim=bond_dim,
			features_in=features_in,
			features_out=features_out,
			std=1e-2,
			init_method='uniform'
		)

		self.peps_var = [[nn.Parameter(torch.from_numpy(i).float(), requires_grad=True) for i in line] for line in peps]
		
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
				peps_line.append(tn.Node(self.peps_var[i][j]))
				input_line.append(tn.Node(inputs[:,i,j]))
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
				index1 = self.index_result[i,j,1]
				index2 = self.index_result[i,j+1,3]
				peps_nodes[i][j][index1] ^ peps_nodes[i][j+1][index2]

		# X Bond
		for j in range(self.ynodes):
			for i in range(self.xnodes-1):
				index1 = self.index_result[i,j,0]
				index2 = self.index_result[i+1,j,2]
				peps_nodes[i][j][index1] ^ peps_nodes[i+1][j][index2]

		# Contract
		# Contract the features
		contracted_peps = []
		for i in range(self.xnodes):
			for j in range(self.ynodes):
				contracted_peps.append(input_nodes[i][j] @ peps_nodes[i][j])

		# Contract the remaining peps (With Auto Mode)
		result = contractors.auto(contracted_peps).tensor
		# print(result[0].item())

		return result.view([10])

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
			out_channels=4,
			kernel_size=(4, 4),
			stride=(4, 4)
		)
		self.dropout = nn.Dropout(0.0)
		
		self.peps_layer = PEPSLayer(
			features_in=4,
			features_out=10,
			xnodes=28//4,
			ynodes=28//4,
			bond_dim=hyper_params['bond_dim']
		)

	def forward(self, inputs):
		inputs = inputs.view(-1, 1, 28, 28)

		out = self.dropout(torch.sigmoid(self.cnn_layer(inputs)))
		out = self.peps_layer(out)

		return out
