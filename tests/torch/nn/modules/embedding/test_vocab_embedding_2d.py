"""
torchrun --nproc_per_node=4 test_vocab_embedding_2d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding2D
from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_batch_2d,
    split_2d,
    gather_2d,
)


def test_vocab_embedding_2d(pc):
    summa_dim = pc.get_world_size(ParallelMode.TENSOR_2D_COL)
    input_ = torch.LongTensor([[0, 1, 6, 3, 8], [5, 2, 7, 4, 9]]).cuda()
    target = torch.randn((2, 5, 8)).cuda()
    dist.broadcast(input_, src=0)
    dist.broadcast(target, src=0)

    vocab_embedding = torch.nn.Embedding(16, 8).cuda()
    w = deepcopy(vocab_embedding.weight.data)

    out = vocab_embedding(input_)
    optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(out, target)
    loss.backward()
    optimizer.step()

    out_update = vocab_embedding(input_)
    input_ = split_batch_2d(input_, summa_dim, parallel_context=pc)
    target = split_2d(target, summa_dim, parallel_context=pc)
    w = split_2d(w, summa_dim, parallel_context=pc)

    vocab_embedding_2d = VocabParallelEmbedding2D(16, 8, parallel_context=pc)
    vocab_embedding_2d.weight.data.copy_(w)

    pout = vocab_embedding_2d(input_)
    optimizer = torch.optim.Adam(vocab_embedding_2d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()

    pout_update = vocab_embedding_2d(input_)
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
        tensor_parallel_size=4,
        tensor_parallel_mode=ParallelMode.TENSOR_2D,
    )
    if pc.get_global_rank() == 0:
        print("Test tests/torch/nn/modules/embedding/test_vocab_embedding_2d.py")
    test_vocab_embedding_2d(pc)
