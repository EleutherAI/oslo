"""
torchrun --nproc_per_node=4 test_vocab_embedding_1d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding1D
from oslo.torch.nn.parallel.tensor_parallel.utils import split_1d


def test_vocab_embedding_1d(pc):
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    world_size = pc.get_world_size(ParallelMode.TENSOR_1D)

    input_ = torch.LongTensor([[0, 1, 6, 3], [5, 2, 7, 9]]).cuda()
    target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
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
    w = split_1d(w, world_size, dim=0, parallel_context=pc)
    vocab_embedding_1d = VocabParallelEmbedding1D(16, 8, parallel_context=pc)
    vocab_embedding_1d.weight.data = w

    pout = vocab_embedding_1d(input_)
    optimizer = torch.optim.Adam(vocab_embedding_1d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()
    pout_update = vocab_embedding_1d(input_)

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
        print("Test tests/torch/nn/modules/embedding/test_vocab_embedding_1d.py")
    test_vocab_embedding_1d(pc)
