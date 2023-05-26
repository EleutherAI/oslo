import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.utils import get_free_port
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.data_parallel.data_parallel import _DistributedDataParallel
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


class DictOutputModel(nn.Module):
    def __init__(self):
        super(DictOutputModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(128, 512)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return {"output1": x1, "output2": x2}


class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(128, 512)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return x1, x2


def run_dist(rank, world_size, model_class):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = model_class()
    ddp_model = _DistributedDataParallel(
        copy.deepcopy(model).to(device), parallel_context
    )
    ddp_model.parallelize()
    model = model.to(device)

    input_data = torch.randn(32, 128).to(device)

    output_normal = model(input_data)

    output_ddp = ddp_model(input_data)

    if isinstance(output_normal, dict):
        for key in output_normal:
            assert torch.allclose(
                output_normal[key], output_ddp[key], rtol=1e-03, atol=1e-03
            ), f"Outputs do not match for key {key}!"
    elif isinstance(output_normal, tuple):
        for i, (out_n, out_f) in enumerate(zip(output_normal, output_ddp)):
            assert torch.allclose(
                out_n, out_f, rtol=1e-03, atol=1e-03
            ), f"Outputs do not match for index {i}!"
    else:
        assert torch.allclose(
            output_normal, output_ddp, rtol=1e-03, atol=1e-03
        ), "Outputs do not match!"


@skip_if_dist_unavailable
def test_ddp():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    models_to_test = [MlpModel, DictOutputModel, MultiOutputModel]

    for model_class in models_to_test:
        print(f"Testing {model_class.__name__}")
        mp.spawn(run_dist, args=(world_size, model_class), nprocs=world_size, join=True)

    print("All tests passed!")
