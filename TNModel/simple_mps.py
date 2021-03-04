import torch
import numpy as np


def simple_side_tensor(bond_dim, phys_dim, std=1e-3, backend='pytorch'):
	if backend == 'pytorch':
		ans = torch.zeros(phys_dim, bond_dim)
		ans[:, 0] = 1.0
		ans = ans.transpose(0, 1)
		return ans + torch.randn_like(ans) * std
	elif backend == 'numpy' or backend == 'tensorflow':
		ans = np.zeros((bond_dim, phys_dim), dtype=np.float32)
		ans[0, :] = 1.0
		return ans + np.random.normal(size=ans.shape) * std
	else:
		raise NotImplementedError()


def simple_middle_tensor(bond_dim, phys_dim, std=1e-3, backend='pytorch'):
	if backend == 'pytorch':
		ans = torch.stack([torch.eye(bond_dim)]*phys_dim, dim=0).transpose(0, 1)
		return ans + torch.randn_like(ans) * std
	elif backend == 'numpy' or backend == 'tensorflow':
		ans = np.stack([np.eye(bond_dim, dtype=np.float32)]*phys_dim, axis=1)
		return ans + np.random.normal(size=ans.shape) * std
	else:
		raise NotImplementedError()


def simple_mps(nodes, bond_dim, phys_dim, std=1e-3, backend='pytorch'):
	assert nodes>=2
	assert len(phys_dim)==nodes

	ans = []
	ans.append(simple_side_tensor(bond_dim, phys_dim[0], std, backend))

	for i in range(1, nodes-1):
		ans.append(simple_middle_tensor(bond_dim, phys_dim[i], std, backend))

	ans.append(simple_side_tensor(bond_dim, phys_dim[-1], std, backend))
	return ans
