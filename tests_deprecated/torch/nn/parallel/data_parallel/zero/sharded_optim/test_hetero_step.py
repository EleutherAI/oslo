import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.utils import get_free_port
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.zero import (
    _HeteroDataParallel,
)
from oslo.torch.nn.parallel.data_parallel.zero import (
    _HeteroOptimizer,
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


def check_param(
    model: _HeteroDataParallel,
    torch_model: torch.nn.Module,
    rtol: float = 1e-3,
    atol: float = 4e-3,
):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        key = key.replace("module.", "")
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device)
        torch.allclose(value.float(), temp_zero_value.float(), rtol=rtol, atol=atol)


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = MlpModel()
    fsdp_model = _HeteroDataParallel(
        copy.deepcopy(model),
        device,
        parallel_context,
        force_outputs_fp32=True,
    )
    fsdp_model.parallelize()
    model = model.to(device)

    torch_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    fsdp_optim = _HeteroOptimizer(
        torch.optim.SGD(fsdp_model.parameters(), lr=0.01),
        fsdp_model,
    )

    input_data = torch.randn(32, 128).to(device)

    torch_optim.zero_grad()
    fsdp_optim.zero_grad()

    output_normal = model(input_data)

    output_fsdp = fsdp_model(input_data)

    output_fsdp.sum().backward()
    output_normal.sum().backward()

    torch_optim.step()
    fsdp_optim.step()

    check_param(fsdp_model, model)
    print(f"Test passed on rank {rank}!")


def test_hetero_step():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(run_dist, args=(world_size,), nprocs=world_size, join=True)
