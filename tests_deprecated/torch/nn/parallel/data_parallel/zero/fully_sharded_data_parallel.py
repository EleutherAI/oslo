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


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = MlpModel()
    fsdp_model = _FullyShardedDataParallel(
        copy.deepcopy(model), device, parallel_context
    )
    model = model.to(device)

    input_data = torch.randn(32, 128).to(device)

    output_normal = model(input_data)

    output_fsdp = fsdp_model(input_data)

    assert torch.allclose(output_normal, output_fsdp), "Outputs do not match!"

    print(f"Test passed on rank {rank}!")


def test_fsdp():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(run_dist, args=(world_size,), nprocs=world_size, join=True)
