import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.utils import get_free_port
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.zero import (
    _HeteroDataParallel,
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


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = MlpModel().to(device)
    fsdp_model = _HeteroDataParallel(
        MlpModel(),
        device,
        parallel_context,
        force_outputs_fp32=True,
    )
    fsdp_model.parallelize()

    zero_dict = fsdp_model.state_dict(only_rank_0=False)
    torch_dict = model.state_dict()

    for key, value in torch_dict.items():
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        assert value.shape == zero_dict[key].shape, "{} shape mismatch {} vs {}".format(
            key, value.shape, zero_dict[key].shape
        )

    fsdp_model.load_state_dict(torch_dict, strict=False)

    input_data = torch.randn(32, 128).to(device)
    torch.allclose(
        model(input_data),
        fsdp_model(input_data),
        atol=1e-3,
        rtol=1e-3,
    )
    print(f"Test passed on rank {rank}!")


def test_state_dict():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(run_dist, args=(world_size,), nprocs=world_size, join=True)
