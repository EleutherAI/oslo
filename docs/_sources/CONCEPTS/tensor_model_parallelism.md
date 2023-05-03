# Concept of Tensor Model Parallelism
- Authors: Kichang Yang, Kevin Ko, Minho Ryu

**Tensor Model Parallelism** makes it possible to train larger models by partitioning the parameter tensors into multiple dimensions.
We support 1D, 2D, 2.5D, and 3D tensor partitioning algorithms which make tensor parallel training more efficient.

## Tensor Parallel Algorithms
- [1D parallel algorithm (same as Megatron-LM)](tp/1d_parallel_algorithm.md)
- [2D parallel (SUMMA) algorithm](tp/2d_parallel_algorithm.md)
- [2.5D parallel (SUMMA-2.5) algorithm](tp/2p5d_parallel_algorithm.md)
- [3D parallel Algorithm](tp/3d_parallel_algorithm.md)

### References
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
- [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)



