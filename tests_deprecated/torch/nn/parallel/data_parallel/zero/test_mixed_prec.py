import os

import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.utils import get_free_port, set_seed
from oslo.torch.nn.parallel.data_parallel.zero import ZeroRedundancyOptimizer
from torch.testing import assert_close

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


def half_close(input, other, loose=False):
    rtol = None
    atol = None
    if loose:
        rtol = 5e-2
        atol = 5e-4

    input = input.detach().half()
    other = other.detach().half()

    assert_close(input, other, rtol=rtol, atol=atol)


def run(parallel_context: ParallelContext):
    local_rank = torch.distributed.get_rank()

    # create model
    model = MlpModel().cuda()
    naive_ddp_model = DDP(model, bucket_cap_mb=0)
    zero_model = copy.deepcopy(model).half()

    # create optimizer
    naive_optimizer = torch.optim.Adam(naive_ddp_model.parameters(), lr=1)
    zero_optimizer = ZeroRedundancyOptimizer(
        torch.optim.Adam(zero_model.parameters(), lr=1),
        parallel_context=parallel_context,
        overlap_communication=True,
    )

    # create data
    set_seed(2021 + local_rank)
    input_data = torch.randn(32, 128).cuda()

    # zero-dp forward
    naive_output = naive_ddp_model(input_data)
    zero_output = zero_model(input_data.half())

    half_close(naive_output, zero_output, loose=True)

    # zero-dp backward
    naive_output.sum().backward()
    zero_output.sum().backward()

    # step
    naive_optimizer.step()
    zero_optimizer.step()

    # check updated param
    for p, zp in zip(naive_ddp_model.parameters(), zero_model.parameters()):
        if zp.grad is not None:
            half_close(p.grad, zp.grad, loose=True)


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)
    run(parallel_context)


@skip_if_dist_unavailable
def test_mixed_prec():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(partial(run_dist, world_size=world_size), nprocs=world_size)
