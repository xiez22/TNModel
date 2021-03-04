# TNModel
Tensor Network models including `MPS`, `1d-SBS` and `CNN-PEPS` for classification with `PyTorch` or `TensorFlow 2` backend and optimizing MPO for Heisenberg XXZ model and 1d-Ising model with Density Matrix Renormalization Group (DMRG). We implement the tensor network algorithms with [Google's tensornetwork](https://github.com/google/TensorNetwork.git).

![](https://img.shields.io/badge/MPS-PyTorch-green.svg) ![](https://img.shields.io/badge/SBS-PyTorch-green.svg) ![](https://img.shields.io/badge/PEPS-PyTorch-yellow.svg)

![](https://img.shields.io/badge/MPS-TensowFlow-green.svg) ![](https://img.shields.io/badge/SBS-TensowFlow-yellow.svg) ![](https://img.shields.io/badge/PEPS-TensorFlow-red.svg)

![](https://img.shields.io/badge/DMRG-XXZ-green.svg) ![](https://img.shields.io/badge/DMRG-Ising-green.svg)
(Green - Finished, Yellow - With Some Problems, Red - Not Finished)

## Installation
- You can choose to use `PyTorch` or `TensorFlow 2` backend.
- Install `tensornetwork>=0.4.4`, `numpy` and `tqdm`.
- Clone this repo.

## Examples
For `PyTorch` and `TensorFlow` backends respectively.

## Notice
__This is a very early version and it is unstable.__

## Reference
- [TensorNetwork for Machine Learning](https://arxiv.org/abs/1906.06329)
- [From Probabilistic Graphical Models to Generalized Tensor Networks for Supervised Learning](https://arxiv.org/abs/1806.05964)
- [Supervised Learning with Projected Entangled Pair States](https://arxiv.org/abs/2009.09932)
