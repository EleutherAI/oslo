from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.chunk import (
    ChunkManager,
)
from oslo.torch.nn.parallel.data_parallel.zero.tensor import (
    DistributedTensorSpec,
    DistributedParameter,
    DistributedTensor,
)
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.utils import get_free_port

import os

import itertools

skip_if_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="dist required"
)

CUDA_MEM_0 = {False: 512, True: 1024}
CUDA_MEM_1 = {False: 0, True: 1024}
CPU_MEM = {True: {True: 0, False: 0}, False: {True: 512, False: 0}}


def exam_chunk_memory(parallel_context, keep_gathered, pin_memory):

    params = [
        DistributedTensor(
            torch.rand(8, 8), spec=DistributedTensorSpec(parallel_context)
        )
        for _ in range(3)
    ]
    config = {2: dict(chunk_size=128, keep_gathered=keep_gathered)}

    chunk_manager = ChunkManager(config)
    assert chunk_manager.total_mem["cpu"] == 0
    assert chunk_manager.total_mem["cuda"] == 0

    for p in params:
        chunk_manager.register_tensor(p, "param", 2, pin_memory=pin_memory)
    chunk_manager.close_all_groups()
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[keep_gathered]

    chunks = chunk_manager.get_chunks(params)

    for chunk in chunks:
        chunk_manager.access_chunk(chunk)
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[True]

    for chunk in chunks:
        chunk_manager.release_chunk(chunk)

    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][pin_memory]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_0[keep_gathered]

    for chunk in chunks:
        chunk_manager.move_chunk(chunk, torch.device("cpu"))
    assert chunk_manager.total_mem["cpu"] == CPU_MEM[keep_gathered][True]
    assert chunk_manager.total_mem["cuda"] == CUDA_MEM_1[keep_gathered]


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    keep_gathered = [True, False]
    pin_memory = [True, False]

    for args in itertools.product(keep_gathered, pin_memory):
        exam_chunk_memory(parallel_context, *args)


@skip_if_dist_unavailable
def test_chunk_manager():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    mp.spawn(partial(run_dist, world_size=world_size), nprocs=world_size)
