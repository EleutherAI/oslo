import os

import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.utils import get_free_port, set_seed
from oslo.torch.nn.parallel.data_parallel.zero import ZeroRedundancyOptimizer

skip_if_no_dist = pytest.mark.skipif(
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


def run(parallel_context: ParallelContext):
    local_rank = torch.distributed.get_rank()
    set_seed(2009)

    # create model
    model = MlpModel().cuda()
    zero1_model = copy.deepcopy(model)
    zero2_model = copy.deepcopy(model)
    naive_ddp_model = DDP(model, bucket_cap_mb=0)

    # create optimizer
    naive_optimizer = torch.optim.Adam(naive_ddp_model.parameters(), lr=1)
    zero1_optimizer = ZeroRedundancyOptimizer(
        torch.optim.Adam(zero1_model.parameters(), lr=1),
        parallel_context=parallel_context,
        overlap_communication=True,
    )
    zero2_optimizer = ZeroRedundancyOptimizer(
        torch.optim.Adam(zero2_model.parameters(), lr=1),
        parallel_context=parallel_context,
        overlap_communication=True,
        partition_grad=True,
    )

    # create data
    set_seed(2021 + local_rank)
    input_data = torch.randn(32, 128).cuda()

    # zero-dp forward
    naive_output = naive_ddp_model(input_data)
    zero1_output = zero1_model(input_data)
    zero2_output = zero2_model(input_data)

    assert torch.equal(naive_output, zero1_output)
    assert torch.equal(zero1_output, zero2_output)

    # zero-dp backward
    naive_output.sum().float().backward()
    zero1_output.sum().float().backward(sync_grad=False)
    zero2_output.sum().float().backward(sync_grad=False)

    for p, z1p, z2p in zip(
        naive_ddp_model.parameters(), zero1_model.parameters(), zero2_model.parameters()
    ):
        if z2p.grad is not None:
            assert torch.equal(p.grad, z1p.grad)
            assert torch.equal(z1p.grad, z2p.grad)

    zero1_optimizer._sync_grad()
    zero2_optimizer._sync_grad()

    # step
    naive_optimizer.step()
    zero1_optimizer.step()
    zero2_optimizer.step()

    # check updated param
    for p, z1p, z2p in zip(
        naive_ddp_model.parameters(), zero1_model.parameters(), zero2_model.parameters()
    ):
        assert torch.equal(p.data, z1p.data)
        assert torch.equal(z1p.data, z2p.data)


def run_dist(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)
    run(parallel_context)


@skip_if_no_dist
def test_integrity():
    world_size = 2
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())

    mp.spawn(partial(run_dist, world_size=world_size), nprocs=world_size)


if __name__ == "__main__":
    test_integrity()
