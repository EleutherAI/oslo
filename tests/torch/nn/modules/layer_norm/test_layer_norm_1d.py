"""
torchrun --nproc_per_node=4 test_layer_norm_1d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import LayerNorm1D


def test_layer_norm_1d(pc):
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

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
    layernorm_1d = LayerNorm1D(hidden_dim, parallel_context=pc)
    layernorm_1d.weight.data = w
    layernorm_1d.bias.data = b

    pout = layernorm_1d(input_)
    optimizer = torch.optim.Adam(layernorm_1d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()
    pout_update = layernorm_1d(input_)

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
        print("Test tests/torch/nn/modules/layer_norm/test_layer_norm_1d.py")
    test_layer_norm_1d(pc)
