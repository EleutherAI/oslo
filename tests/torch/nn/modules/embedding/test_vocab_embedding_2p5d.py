"""
torchrun --nproc_per_node=8 test_vocab_embedding_2p5d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_batch_2p5d,
    split_2p5d,
    gather_2p5d,
)
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding2p5D


def test_vocab_embedding_2p5d(pc):
    batch_size = 2
    seq_len = 5
    num_embeddings = 16
    embedding_dim = 8
    tesseract_dim = pc.get_world_size(ParallelMode.TENSOR_2P5D_COL)

    input_ = torch.LongTensor([[0, 1, 6, 3, 8], [5, 2, 7, 4, 9]]).cuda()
    target = torch.randn((batch_size, seq_len, embedding_dim)).cuda()
    dist.broadcast(input_, src=0)
    dist.broadcast(target, src=0)

    vocab_embedding = torch.nn.Embedding(num_embeddings, embedding_dim).cuda()
    w = deepcopy(vocab_embedding.weight.data)

    out = vocab_embedding(input_)
    optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(out, target)
    loss.backward()
    optimizer.step()

    out_update = vocab_embedding(input_)
    input_ = split_batch_2p5d(input_, tesseract_dim, parallel_context=pc)
    target = split_2p5d(target, tesseract_dim, parallel_context=pc)
    w = split_2p5d(w, tesseract_dim, parallel_context=pc)

    vocab_embedding_2p5d = VocabParallelEmbedding2p5D(
        num_embeddings, embedding_dim, parallel_context=pc
    )
    vocab_embedding_2p5d.weight.data.copy_(w)

    pout = vocab_embedding_2p5d(input_)
    optimizer = torch.optim.Adam(vocab_embedding_2p5d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()

    pout_update = vocab_embedding_2p5d(input_)
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
        print("Test tests/torch/nn/modules/embedding/test_vocab_embedding_2p5d.py")
    test_vocab_embedding_2p5d(pc)
