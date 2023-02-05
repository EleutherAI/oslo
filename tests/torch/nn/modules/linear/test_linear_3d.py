"""
torchrun --nproc_per_node=4 test_linear_3d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear3D
from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_input_3d,
    split_weight_3d,
    split_bias_3d,
    gather_output_3d,
)


def test_linear_3d(pc):
    batch_size = 4
    seq_len = 2
    input_dim = 4
    hidden_dim = 8
    cubic_dim = pc.get_world_size(ParallelMode.TENSOR_3D_INPUT)

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
    input_ = split_input_3d(input_, cubic_dim, parallel_context=pc)
    ptarget = split_input_3d(target, cubic_dim, parallel_context=pc)

    w = split_weight_3d(w, cubic_dim, parallel_context=pc)
    b = split_bias_3d(b, cubic_dim, parallel_context=pc)

    linear_3d = Linear3D(input_dim, hidden_dim, parallel_context=pc).cuda()
    linear_3d.weight.data.copy_(w)
    linear_3d.bias.data.copy_(b)

    pout = linear_3d(input_)
    optimizer = torch.optim.Adam(linear_3d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, ptarget)
    loss.backward()
    optimizer.step()

    pout_update = linear_3d(input_)
    pout = gather_output_3d(pout, cubic_dim, parallel_context=pc)
    pout_update = gather_output_3d(pout_update, cubic_dim, parallel_context=pc)

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
        tensor_parallel_size=8,
        tensor_parallel_mode=ParallelMode.TENSOR_3D,
    )
    if pc.get_global_rank() == 0:
        print("Test tests/torch/nn/modules/linear/test_linear_3d.py")
    test_linear_3d(pc)
