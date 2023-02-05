"""
torchrun --nproc_per_node=8 test_layer_norm_3d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo import ParallelMode
from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import LayerNorm3D
from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_input_3d,
    split_layernorm_3d,
    split_bias_3d,
    gather_output_3d,
)


def test_layer_norm_3d(pc):
    batch_size = 4
    seq_len = 2
    hidden_dim = 8

    cubic_dim = pc.get_world_size(ParallelMode.TENSOR_3D_INPUT)
    input_ = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
    target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()

    dist.broadcast(input_, src=0)
    dist.broadcast(target, src=0)

    layernorm = torch.nn.LayerNorm(hidden_dim).cuda()
    w = deepcopy(layernorm.weight.data)
    b = deepcopy(layernorm.bias.data)

    out = layernorm(input_)
    optimizer = torch.optim.Adam(layernorm.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(out, target)
    loss.backward()
    optimizer.step()

    out_update = layernorm(input_)
    dist.barrier()

    input_ = split_input_3d(input_, cubic_dim, parallel_context=pc)
    target = split_input_3d(target, cubic_dim, parallel_context=pc)
    w = split_layernorm_3d(w, cubic_dim, parallel_context=pc)
    b = split_bias_3d(b, cubic_dim, parallel_context=pc)

    layernorm_3d = LayerNorm3D(hidden_dim, parallel_context=pc)
    layernorm_3d.weight.data.copy_(w)
    layernorm_3d.bias.data.copy_(b)

    pout = layernorm_3d(input_)
    optimizer = torch.optim.Adam(layernorm_3d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()

    pout_update = layernorm_3d(input_)
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
        print("Test tests/torch/nn/modules/layer_norm/test_layer_norm_3d.py")
    test_layer_norm_3d(pc)
