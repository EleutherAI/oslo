"""
torchrun --nproc_per_node=4 test_vocab_parallel_cross_entropy_1d.py
"""

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelCrossEntropyLoss1D
from oslo.torch.nn.parallel.tensor_parallel.utils import split_1d


def test_vocab_parallel_cross_entropy_1d(pc):
    criterion_master = torch.nn.CrossEntropyLoss()
    criterion = VocabParallelCrossEntropyLoss1D(parallel_context=pc)

    batch_size = 2
    seq_len = 4
    num_classes = 8
    world_size = pc.get_world_size(ParallelMode.TENSOR_1D)

    out_master = torch.randn(batch_size, seq_len, num_classes).cuda()
    target = torch.randint(
        num_classes, size=(batch_size, seq_len), dtype=torch.long
    ).cuda()
    dist.broadcast(out_master, src=0)
    dist.broadcast(target, src=0)

    out = split_1d(out_master.clone(), world_size, dim=-1, parallel_context=pc)
    out = out.clone()
    out.requires_grad = True
    loss = criterion(out, target)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(
        out_master.view(-1, out_master.size(-1)), target.view(-1)
    )

    if pc.get_global_rank() == 0:
        print("> Test forward...", end="")
    assert torch.allclose(loss_master, loss)
    if pc.get_global_rank() == 0:
        print("OK")

    loss_master.backward()
    loss.backward()

    grad_master = out_master.grad
    grad_master = split_1d(grad_master, world_size, dim=-1, parallel_context=pc)
    grad = out.grad

    if pc.get_global_rank() == 0:
        print("> Test backward...", end="")
    assert torch.allclose(grad_master, grad)
    if pc.get_global_rank() == 0:
        print("OK")


if __name__ == "__main__":
    pc = ParallelContext.from_torch(tensor_parallel_size=4)
    if pc.get_global_rank() == 0:
        print(
            "Test tests/torch/nn/modules/linear/test_vocab_parallel_cross_entropy_1d.py"
        )
    test_vocab_parallel_cross_entropy_1d(pc)
