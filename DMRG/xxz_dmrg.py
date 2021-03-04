"""
DMRG for XXZ model.
"""

from typing import Type, Text
import tensornetwork as tn
import numpy as np

tn.set_default_backend('pytorch')

def initialize_spin_mps(N: int, D: int, dtype: Type[np.number]):
	return tn.FiniteMPS.random([2] * N, [D] * (N - 1), dtype=dtype)


def initialize_XXZ_mpo(Jz: np.ndarray, Jxy: np.ndarray, Bz: np.ndarray, dtype: Type[np.number]):
	return tn.FiniteXXZ(Jz, Jxy, Bz, dtype=dtype)

def run_twosite_dmrg(N: int, D: int, dtype: Type[np.number], Jz: np.ndarray, Jxy: np.ndarray, Bz: np.ndarray, num_sweeps: int):
	mps = initialize_spin_mps(N, 32, dtype)
	mpo = initialize_XXZ_mpo(Jz, Jxy, Bz, dtype)
	dmrg = tn.FiniteDMRG(mps, mpo)
	result = dmrg.run_two_site(
					max_bond_dim=D, num_sweeps=num_sweeps, num_krylov_vecs=10, verbose=1)

	return result, mps


if __name__ == '__main__':
	num_sites, bond_dim, datatype = 20, 16, np.float64
	jz = np.ones(num_sites - 1)
	jxy = np.ones(num_sites - 1)
	bz = np.zeros(num_sites)
	n_sweeps = 5
	print(f'\nrunning DMRG for XXZ model...')
	
	energy, mps = run_twosite_dmrg(num_sites, bond_dim, datatype, jxy, jz, bz, num_sweeps=n_sweeps)

	print(f'\nFinished. Energy: {energy}')
	# print(mps.tensors)
