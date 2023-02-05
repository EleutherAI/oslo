"""
torchrun --nproc_per_node=8 test_layer_norm_2p5d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo import ParallelMode
from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import LayerNorm2p5D
from oslo.torch.nn.parallel.tensor_parallel.utils import (
    gather_2p5d,
    split_2p5d,
    split_layernorm_2p5d,
    split_bias_2p5d,
)


def test_layer_norm_2p5d(pc):
    batch_size = 2
    seq_len = 2
    hidden_dim = 8

    tesseract_dim = pc.get_world_size(ParallelMode.TENSOR_2P5D_COL)
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

    input_ = split_2p5d(input_, tesseract_dim, parallel_context=pc)
    target = split_2p5d(target, tesseract_dim, parallel_context=pc)

    w = split_layernorm_2p5d(w, tesseract_dim, parallel_context=pc)
    b = split_bias_2p5d(b, tesseract_dim, parallel_context=pc)

    layernorm_2p5d = LayerNorm2p5D(hidden_dim, parallel_context=pc)
    layernorm_2p5d.weight.data.copy_(w)
    layernorm_2p5d.bias.data.copy_(b)

    pout = layernorm_2p5d(input_)
    optimizer = torch.optim.Adam(layernorm_2p5d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()

    pout_update = layernorm_2p5d(input_)
    pout = gather_2p5d(pout, tesseract_dim, parallel_context=pc)
    pout_update = gather_2p5d(pout_update, tesseract_dim, parallel_context=pc)

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
        tensor_parallel_depth=2,
        tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    )
    if pc.get_global_rank() == 0:
        print("Test tests/torch/nn/modules/layer_norm/test_layer_norm_2p5d.py")
    test_layer_norm_2p5d(pc)
