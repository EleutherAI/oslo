import os
import torch.multiprocessing as mp

import torch
from oslo.torch.nn.parallel.data_parallel import DistributedDataParallel as DDP
from torch import nn
from torch import optim
import torch.distributed as dist

from oslo.torch.distributed.parallel_context import ParallelContext


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train(rank, world_size):
    print(f"Running oslo DDP example on rank {rank}.")
    setup(rank, world_size)
    parallel_context = ParallelContext.from_torch(data_parallel_size=world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, parallel_context)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.zeros(20, 10).to(rank))
    labels = torch.zeros(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print(outputs)
    cleanup()


def main(world_size):
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main(2)
