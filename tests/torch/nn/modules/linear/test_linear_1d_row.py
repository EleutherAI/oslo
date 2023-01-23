"""
torchrun --nproc_per_node=4 test_linear_1d_row.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import RowLinear1D
from oslo.torch.nn.parallel.tensor_parallel.utils import split_1d


def test_linear_1d_row(pc):
    batch_size = 2
    seq_len = 4
    input_dim = 4
    hidden_dim = 8
    world_size = pc.get_world_size(ParallelMode.TENSOR_1D)

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
    input_ = split_1d(input_, world_size, dim=-1, parallel_context=pc)
    w = split_1d(w, world_size, dim=-1, parallel_context=pc)

    row_linear = RowLinear1D(input_dim, hidden_dim, parallel_context=pc).cuda()
    row_linear.weight.data.copy_(w)
    row_linear.bias.data.copy_(b)

    pout = row_linear(input_)
    optimizer = torch.optim.Adam(row_linear.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()
    pout_update = row_linear(input_)

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
    pc = ParallelContext.from_torch(tensor_parallel_size=4)
    if pc.get_global_rank() == 0:
        print("Test tests/torch/nn/modules/linear/test_linear_1d_row.py")
    test_linear_1d_row(pc)
