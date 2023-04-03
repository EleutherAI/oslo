import math
import torch
import torch.distributed as dist
import pytest
import torch.multiprocessing as mp
from oslo.torch.utils import get_free_port
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.zero.tensor import DistributedSpecManager
from oslo.torch.nn.parallel.data_parallel.zero.tensor.distributed_spec import ShardSpec, ReplicaSpec
from functools import partial

import os


skip_if_multi_dim_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="dist required"
)


def _manipulate_spec(parallel_context):
    rank = dist.get_rank()
    size = dist.get_world_size()
    depth = int(math.sqrt(size))
    assert depth == math.sqrt(size)
    x = torch.rand(8, 8).cuda()
    old_dist_spec = ReplicaSpec()
    row_spec = ShardSpec([0], [size])
    col_spec = ShardSpec([-1], [size])
    mat_spec = ShardSpec([0, 1], [depth, depth])
    row_shard = DistributedSpecManager._shard_as(x, old_dist_spec, row_spec, parallel_context)
    assert torch.equal(x.chunk(size, 0)[rank], row_shard)
    assert torch.equal(x, DistributedSpecManager._gather(row_shard, row_spec, parallel_context))
    col_shard = DistributedSpecManager._all_to_all(row_shard, row_spec, col_spec, parallel_context)
    assert torch.equal(x.chunk(size, -1)[rank], col_shard)
    assert torch.equal(x, DistributedSpecManager._gather(col_shard, col_spec, parallel_context))
    mat_shard = DistributedSpecManager._shard_as(x, old_dist_spec, mat_spec, parallel_context)
    assert torch.equal(x.chunk(depth, 0)[rank // depth].chunk(depth, 1)[rank % depth], mat_shard)
    assert torch.equal(x, DistributedSpecManager._gather(mat_shard, mat_spec, parallel_context))


def _check_mem(parallel_context):
    size = dist.get_world_size()
    assert torch.cuda.memory_allocated() == 0
    x = torch.rand(32, 32).cuda()
    orig_mem = x.numel() * x.element_size()
    assert torch.cuda.memory_allocated() == orig_mem
    old_dist_spec = ReplicaSpec()
    row_spec = ShardSpec([0], [size])
    x.data = DistributedSpecManager._shard_as(x, old_dist_spec, row_spec, parallel_context)
    assert x.size(0) == 32 // size and x.size(1) == 32
    assert torch.cuda.memory_allocated() == orig_mem // size
    x.data = DistributedSpecManager._gather(x, row_spec, parallel_context)
    assert torch.cuda.memory_allocated() == orig_mem


def run_dist_tests(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(tensor_parallel_size=world_size)
    _check_mem(parallel_context)
    _manipulate_spec(parallel_context)


@skip_if_multi_dim_dist_unavailable
def test_dist_cases(world_size: int = 4):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    run_func = partial(run_dist_tests, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)