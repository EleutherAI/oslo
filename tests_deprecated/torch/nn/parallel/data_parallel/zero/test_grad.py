import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.utils import get_free_port
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.zero.fully_sharded_data_parallel import (
    _FullyShardedDataParallel,
)
import copy
import pytest

skip_if_dist_unavailable = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="dist required"
)


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def check_grad(model: _FullyShardedDataParallel, torch_model: torch.nn.Module):
    chunk_manager = model.chunk_manager
    param_list = [p for p in model.parameters()]
    chunk_list = chunk_manager.get_chunks(param_list)
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for (p0, p1) in zip(model.parameters(), torch_model.parameters()):
        torch.allclose(p0.float(), p1.grad, rtol=1e-3, atol=5e-5)


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = MlpModel()
    fsdp_model = _FullyShardedDataParallel(
        copy.deepcopy(model),
        device,
        parallel_context,
        force_outputs_fp32=True,
    )
    fsdp_model.parallelize()
    model = model.to(device)

    input_data = torch.randn(32, 128).to(device)

    output_normal = model(input_data)

    output_fsdp = fsdp_model(input_data)

    output_fsdp.sum().backward()
    output_normal.sum().backward()

    check_grad(fsdp_model, model)

    print(f"Test passed on rank {rank}!")


def test_grad():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(run_dist, args=(world_size,), nprocs=world_size, join=True)
