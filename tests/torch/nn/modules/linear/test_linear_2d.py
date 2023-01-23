"""
torchrun --nproc_per_node=4 test_linear_2d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2D
from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_2d,
    gather_2d,
    split_bias_2d,
)


def test_linear_2d(pc):
    batch_size = 2
    seq_len = 2
    input_dim = 4
    hidden_dim = 8
    summa_dim = pc.get_world_size(ParallelMode.TENSOR_2D_COL)

    input_ = torch.randn((batch_size, seq_len, input_dim)).cuda()
    target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
    dist.broadcast(input_, src=0)
    dist.broadcast(target, src=0)

    linear = torch.nn.Linear(input_dim, hidden_dim).cuda()
    w = deepcopy(linear.weight.data)
    b = deepcopy(linear.bias.data)

    out = linear(input_)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(out, target)
    loss.backward()
    optimizer.step()
    out_update = linear(input_)

    input_ = split_2d(input_, summa_dim, parallel_context=pc)
    ptarget = split_2d(target, summa_dim, parallel_context=pc)
    w = split_2d(w, summa_dim, parallel_context=pc)
    b = split_bias_2d(b, summa_dim, parallel_context=pc)

    linear_2d = Linear2D(input_dim, hidden_dim, parallel_context=pc).cuda()
    linear_2d.weight.data.copy_(w)
    linear_2d.bias.data.copy_(b)

    pout = linear_2d(input_)
    optimizer = torch.optim.Adam(linear_2d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, ptarget)
    loss.backward()
    optimizer.step()

    pout_update = linear_2d(input_)
    pout = gather_2d(pout, summa_dim, parallel_context=pc)
    pout_update = gather_2d(pout_update, summa_dim, parallel_context=pc)

    if pc.get_global_rank() == 0:
        print("> Test forward...", end="")
    assert torch.allclose(out, pout)
    if pc.get_global_rank() == 0:
        print("OK")

    if pc.get_global_rank() == 0:
        print("> Test backward...", end="")
    assert torch.allclose(out_update, pout_update)
    if pc.get_global_rank() == 0:
        print("OK")


if __name__ == "__main__":
    pc = ParallelContext.from_torch(
        tensor_parallel_size=4, tensor_parallel_mode=ParallelMode.TENSOR_2D
    )
    if pc.get_global_rank() == 0:
        print("Test tests/torch/nn/modules/linear/test_linear_2d.py")
    test_linear_2d(pc)
