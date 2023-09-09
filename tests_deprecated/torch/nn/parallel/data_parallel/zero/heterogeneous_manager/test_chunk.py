from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from oslo.torch.nn.parallel.data_parallel.zero.hetero.chunk import (
    TensorState,
    Chunk,
)
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.utils import get_free_port
from oslo.torch.nn.parallel.data_parallel.zero.hetero.utils import get_current_device

import itertools

skip_if_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="dist required"
)


def dist_sum(x):
    temp = torch.tensor([x], device=get_current_device())
    dist.all_reduce(temp)
    return temp.item()


def add_param(param_list, param_cp_list, *args, **kwargs):
    param = torch.randn(*args, **kwargs)
    param_list.append(param)
    param_cp_list.append(param.clone())


def check_euqal(param, param_cp):
    if param.device != param_cp.device:
        temp = param.data.to(param_cp.device)
    else:
        temp = param.data
    return torch.equal(temp, param_cp.data)


def exam_chunk_basic(parallel_context, init_device, keep_gathered, pin_memory):
    world_size = torch.distributed.get_world_size()
    my_chunk = Chunk(
        chunk_size=1024,
        parallel_context=parallel_context,
        dtype=torch.float32,
        init_device=init_device,
        keep_gathered=keep_gathered,
        pin_memory=pin_memory,
        cpu_shard_init=True,
    )

    param_list = []
    param_cp_list = []

    add_param(param_list, param_cp_list, 8, 8, 8, device="cuda")
    add_param(param_list, param_cp_list, 4, 4)
    add_param(param_list, param_cp_list, 4, 8, 2, device="cuda")
    add_param(param_list, param_cp_list, 1, 1, 5)

    for param in param_list:
        my_chunk.append_tensor(param)
    assert my_chunk.utilized_size == 597
    for param, param_cp in zip(param_list, param_cp_list):
        check_euqal(param, param_cp)
    my_chunk.close_chunk()

    if keep_gathered is False:
        assert my_chunk.cpu_shard.size(0) == 1024 // world_size
        assert my_chunk.device_type == "cpu"
        assert my_chunk.can_move
        my_chunk.shard_move(get_current_device())
    else:
        assert my_chunk.cuda_global_chunk.size(0) == 1024
        assert my_chunk.device_type == "cuda"
        assert not my_chunk.can_move

    assert dist_sum(my_chunk.valid_end) == my_chunk.utilized_size
    flag = my_chunk.has_inf_or_nan
    assert not flag, "has_inf_or_nan is {}".format(flag)

    my_chunk.access_chunk()
    assert my_chunk.device_type == "cuda"
    for param, param_cp in zip(param_list, param_cp_list):
        check_euqal(param, param_cp)

    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 4
    my_chunk.tensor_trans_state(param_list[0], TensorState.COMPUTE)
    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 3
    assert my_chunk.tensor_state_cnter[TensorState.COMPUTE] == 1
    assert not my_chunk.can_release

    for param in param_list:
        my_chunk.tensor_trans_state(param, TensorState.COMPUTE)
        my_chunk.tensor_trans_state(param, TensorState.HOLD_AFTER_BWD)
        my_chunk.tensor_trans_state(param, TensorState.READY_FOR_REDUCE)

    assert my_chunk.tensor_state_cnter[TensorState.READY_FOR_REDUCE] == 4
    assert my_chunk.can_reduce
    my_chunk.reduce()
    assert my_chunk.tensor_state_cnter[TensorState.HOLD] == 4

    if keep_gathered is False:
        assert my_chunk.cuda_shard.size(0) == 1024 // world_size
        assert my_chunk.device_type == "cuda"
        assert my_chunk.can_move
    else:
        assert my_chunk.cuda_global_chunk.size(0) == 1024
        assert my_chunk.device_type == "cuda"
        assert not my_chunk.can_move


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    init_device = [None, torch.device("cpu")]
    keep_gathered = [True, False]
    pin_memory = [True, False]

    for args in itertools.product(init_device, keep_gathered, pin_memory):
        exam_chunk_basic(parallel_context, *args)


@skip_if_dist_unavailable
def test_chunk_function():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    mp.spawn(partial(run_dist, world_size=world_size), nprocs=world_size)
