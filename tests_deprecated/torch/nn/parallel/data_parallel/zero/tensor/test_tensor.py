import torch
import pytest
from numpy import allclose

from oslo.torch.utils import get_free_port
from oslo.torch.nn.parallel.data_parallel.zero.tensor.distributed_spec import ShardSpec, ReplicaSpec
from oslo.torch.nn.parallel.data_parallel.zero.tensor.distributed_tensor_spec import DistributedTensorSpec
from oslo.torch.nn.parallel.data_parallel.zero.tensor.distributed_tensor import DistributedTensor
from oslo.torch.distributed.parallel_mode import ParallelMode
import torch.multiprocessing as mp
from functools import partial
import os
from oslo.torch.distributed.parallel_context import ParallelContext

skip_if_multi_dim_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="dist required"
)


def _run_tensor_shard_init(parallel_context):
    t_ref = torch.randn(4, 5)
    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    shard_attr = ShardSpec(dims=[0], num_partitions=[tp_world_size])
    tensor_spec = DistributedTensorSpec(parallel_context, dist_attr=shard_attr)
    t = DistributedTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    t.set_dist_spec(ReplicaSpec())
    assert t.shape == torch.Size((4 * tp_world_size, 5)), f"{t.shape} vs ({4 * tp_world_size, 5})"


def _run_tensor_replicated_init(parallel_context):
    dp_world_size = parallel_context.get_world_size(ParallelMode.DATA)
    t_ref = torch.randn(4 * dp_world_size, 5)
    spec = DistributedTensorSpec(parallel_context)
    t = DistributedTensor.from_torch_tensor(t_ref.clone(), spec)
    assert t.shape == torch.Size((4 * dp_world_size, 5)), f"{t.shape}"


#### Test Distributed init a DistributedTensor
def _run_view(parallel_context):
    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    t_ref = torch.randn(4, 5)
    t = DistributedTensor.from_torch_tensor(
        t_ref, DistributedTensorSpec(parallel_context, dist_attr=ShardSpec(dims=[0], num_partitions=[tp_world_size])))

    assert t.size_global()[0] == 4 * tp_world_size
    assert t.size_global(1) == 5
    assert t.size_global() == torch.Size([4 * tp_world_size, 5])

    t.set_dist_spec(ReplicaSpec())
    t = t.view(4 * 5 * tp_world_size)
    assert t.shape == torch.Size([4 * 5 * tp_world_size])


def _run_tensor_indexing(parallel_context):
    spec = DistributedTensorSpec(parallel_context, ReplicaSpec())
    torch_t = torch.randn(2, 3)
    dist_t = DistributedTensor(torch_t, spec)
    assert allclose(torch_t[:, 1], dist_t[:, 1])


def _run_operand(parallel_context):
    t_ref = torch.randn(4, 5)
    t = DistributedTensor.from_torch_tensor(t_ref.clone(), DistributedTensorSpec(parallel_context))
    t_ref_res = t_ref + t_ref
    t_res = t + t
    assert isinstance(t_res, DistributedTensor)
    assert torch.allclose(t_ref_res, t_res)
    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    t = DistributedTensor.from_torch_tensor(t_ref.clone(), DistributedTensorSpec(parallel_context))
    t.set_dist_spec(ShardSpec([0], [tp_world_size]))
    t_new = torch.zeros_like(t)
    assert isinstance(t_new, DistributedTensor)
    assert t_new.is_sharded()


def _run_wrapped_tensor_func(parallel_context):
    t_ref = torch.randn(4, 5)
    t = DistributedTensor.from_torch_tensor(t_ref.clone(), DistributedTensorSpec(parallel_context))
    # non-func attr
    assert t.is_cuda == t_ref.is_cuda
    # return 1 torch.Tensor
    t_abs = t.abs()
    assert isinstance(t_abs, DistributedTensor) and torch.equal(t_abs, t_ref.abs())
    # return 1 non-torch.Tensor
    assert t.dim() == t_ref.dim()
    # return >1 torch.Tensor
    assert isinstance(t, DistributedTensor)
    t_split1, t_split2 = t.split(2)
    assert isinstance(t_split1, DistributedTensor) and isinstance(t_split2, DistributedTensor), f"{type(t_split1)} {type(t_split2)}"


def _run_redistributed(parallel_context):
    global_world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
    if global_world_size != 4:
        return
    spec1 = DistributedTensorSpec(parallel_context)
    t1 = DistributedTensor.from_torch_tensor(torch.randn(2, 3, 4), spec1)
    t1 = t1.redistribute(ShardSpec([0], [2]))
    assert t1.is_sharded()
    t1 = t1.redistribute(ShardSpec([-1], [4]))
    assert t1.is_sharded()
    t1 = t1.redistribute(ReplicaSpec())
    assert t1.is_replicate(parallel_context)


def _run_set_tensor_spec(parallel_context):
    global_world_size = parallel_context.get_world_size(ParallelMode.GLOBAL)
    if global_world_size != 4:
        return
    tp_world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
    parallel_context = ParallelContext.from_torch(tensor_parallel_size=tp_world_size)
    spec1 = DistributedTensorSpec(parallel_context)
    t1 = DistributedTensor.from_torch_tensor(torch.randn(2, 3, 4), spec1)
    dist_spec2 = ShardSpec([-1], [global_world_size])
    assert t1.is_replicate()
    t1.set_dist_spec(dist_spec2)
    assert t1.is_shard_1dcol()


def run_dist_tests(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=2, tensor_parallel_size=world_size//2)
    
    _run_tensor_shard_init(parallel_context)
    _run_tensor_replicated_init(parallel_context)
    _run_view(parallel_context)
    _run_tensor_indexing(parallel_context)
    _run_operand(parallel_context)
    _run_wrapped_tensor_func(parallel_context)
    _run_redistributed(parallel_context)
    _run_set_tensor_spec(parallel_context)


@skip_if_multi_dim_dist_unavailable
def test_dist_cases(world_size: int = 4):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    run_func = partial(run_dist_tests, world_size=world_size)
    mp.spawn(run_func, nprocs=world_size)