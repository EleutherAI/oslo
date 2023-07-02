import os

import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import oslo
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.utils import get_free_port, set_seed
from oslo.torch.nn.parallel.data_parallel.zero import ZeroRedundancyOptimizer
from oslo.torch.nn.parallel import TensorParallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

skip_if_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="dist required"
)


def assert_shard_close(
    tensor: torch.Tensor,
    shard: torch.Tensor,
    rank: int,
    world_size: int,
    rtol: float = 1e-3,
    atol: float = 1e-1,
):
    assert tensor.ndim == shard.ndim
    if tensor.shape == shard.shape:
        return torch.allclose(tensor, shard, rtol=rtol, atol=atol)
    else:
        dims_not_eq = torch.nonzero(
            torch.tensor(tensor.shape) != torch.tensor(shard.shape)
        )
        if dims_not_eq.numel() == 1:
            dim = dims_not_eq.item()
            return torch.allclose(
                tensor.chunk(world_size, dim)[rank], shard, rtol=rtol, atol=atol
            )
        else:
            raise NotImplementedError


def run(parallel_context: ParallelContext):
    local_rank = torch.distributed.get_rank()

    # create model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    hybrid_model = TensorParallel(
        copy.deepcopy(model), parallel_context=parallel_context
    )
    oslo.ready(hybrid_model, parallel_context)
    zero_model = model.cuda()

    # create optimizer
    hybrid_optimizer = ZeroRedundancyOptimizer(
        torch.optim.Adam(hybrid_model.parameters(), lr=1e-2),
        parallel_context=parallel_context,
    )
    zero_optimizer = ZeroRedundancyOptimizer(
        torch.optim.Adam(zero_model.parameters(), lr=1e-2),
        parallel_context=parallel_context,
    )

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # create data
    set_seed(2021 + local_rank)
    input_text = ["This is a sample text."] * 32
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    labels = torch.randint(0, model.config.num_labels, (32,)).long().cuda()

    # zero-dp forward
    hybrid_output = hybrid_model(**inputs, labels=labels).loss
    zero_output = zero_model(**inputs, labels=labels).loss

    assert torch.allclose(hybrid_output, zero_output)

    # zero-dp backward
    hybrid_output.backward()
    zero_output.backward()

    # step
    hybrid_optimizer.step()
    zero_optimizer.step()

    # check updated param
    for hp, zp in zip(hybrid_model.parameters(), zero_model.parameters()):
        assert assert_shard_close(
            zp.data, hp.data, local_rank, torch.distributed.get_world_size()
        )


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(tensor_parallel_size=world_size)
    run(parallel_context)


@skip_if_dist_unavailable
def test_hybrid():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(partial(run_dist, world_size=world_size), nprocs=world_size)
