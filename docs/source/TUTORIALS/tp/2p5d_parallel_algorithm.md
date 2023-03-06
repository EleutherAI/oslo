# 2.5D parallel (SUMMA-2.5) algorithm

[https://arxiv.org/pdf/2105.14500.pdf](https://arxiv.org/pdf/2105.14500.pdf)

The 2D parallelism algorithm has lower memory cost than 1D parallelism, but can increase communication. To address this, a 2.5D tensor parallelism algorithm based on 2.5D SUMMA was proposed, which reduces communication by using more devices. The algorithm involves splitting the input and weight matrices into smaller sub-matrices, and applying the SUMMA algorithm to each sub-matrix. The output is then computed by combining the results of each sub-matrix. This approach can be used for linear layers and other tensor operations.

## Usage

Using `ParallelMode.TENSOR_2p5D` as a parameter of `tensor_parallel_mode`. Also, you should `tp_depth` to more than **1. If you use 1, It is identical to 2D algorithm.**

```python
# model = defined in section 2.2

tp_size = 4
tp_depth = 1

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2p5D,
    tensor_parallel_depth=tp_depth,
)
oslo.ready(model, parallel_context)
```