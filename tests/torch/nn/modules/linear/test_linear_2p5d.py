"""
torchrun --nproc_per_node=4 test_linear_2p5d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2p5D
from oslo.torch.nn.parallel.tensor_parallel._2p5d._ops import split_2d, gather_2d


def test_linear_2p5d(pc):
    batch_size = 4
    seq_len = 2
    input_dim = 4
    hidden_dim = 8
    tesseract_dim = pc.get_world_size(ParallelMode.TENSOR_2P5D_COL)

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
    input_ = split_2d(pc, input_, tesseract_dim)
    ptarget = split_2d(pc, target, tesseract_dim)

    w = split_2d(pc, w, tesseract_dim, col_first=False)
    b = b.chunk(tesseract_dim, dim=0)[pc.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)]

    linear_2p5d = Linear2p5D(4, 4, parallel_context=pc).cuda()
    linear_2p5d.weight.data.copy_(w)
    linear_2p5d.bias.data.copy_(b)

    pout = linear_2p5d(input_)
    optimizer = torch.optim.Adam(linear_2p5d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, ptarget)
    loss.backward()
    optimizer.step()

    pout_update = linear_2p5d(input_)
    pout = gather_2d(pc, pout, tesseract_dim, False)
    pout_update = gather_2d(pc, pout_update, tesseract_dim, False)

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
        print("Test tests/torch/nn/modules/linear/test_linear_2p5d.py")
    test_linear_2p5d(pc)
