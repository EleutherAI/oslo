"""
torchrun --nproc_per_node=8 test_vocab_embedding_3d.py
"""
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.torch.nn.parallel.tensor_parallel.utils import (
    split_batch_3d,
    split_input_3d,
    split_weight_3d,
    gather_output_3d,
)
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding3D


def test_vocab_embedding_3d(pc):
    batch_size = 4
    seq_len = 5
    num_embeddings = 16
    embedding_dim = 8
    cubic_dim = pc.get_world_size(ParallelMode.TENSOR_3D_INPUT)

    input_ = torch.LongTensor(
        [[0, 1, 6, 13, 8], [5, 12, 7, 4, 9], [5, 2, 7, 15, 4], [14, 2, 8, 7, 9]]
    ).cuda()
    target = torch.randn((batch_size, seq_len, embedding_dim)).cuda()
    dist.broadcast(input_, src=0)
    dist.broadcast(target, src=0)

    embedding = torch.nn.Embedding(num_embeddings, embedding_dim).cuda()
    w = deepcopy(embedding.weight.data)

    out = embedding(input_)
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(out, target)
    loss.backward()
    optimizer.step()

    out_update = embedding(input_)
    input_ = split_batch_3d(input_, cubic_dim, parallel_context=pc)
    target = split_input_3d(target, cubic_dim, parallel_context=pc)
    w = split_weight_3d(w, cubic_dim, parallel_context=pc)

    vocab_embedding_3d = VocabParallelEmbedding3D(
        num_embeddings, embedding_dim, parallel_context=pc
    )
    vocab_embedding_3d.weight.data.copy_(w)

    pout = vocab_embedding_3d(input_)
    optimizer = torch.optim.Adam(vocab_embedding_3d.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()(pout, target)
    loss.backward()
    optimizer.step()

    pout_update = vocab_embedding_3d(input_)
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
        print("Test tests/torch/nn/modules/embedding/test_vocab_embedding_3d.py")
    test_vocab_embedding_3d(pc)
