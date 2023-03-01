import os
import random
import time
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader

import oslo
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel
from oslo.torch.utils.extensions import ready_torch

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

os.environ["TRANSFORMER_CACHE"] = "/fsx/seokung/.cache"

# torch.set_printoptions(threshold=10_000)

world_size = 1
expert_parallel_size = 1
num_experts = {
    (0,): 4,
    (1,): 4,
    (2,): 4,
    (3,): 4,
    (4,): 4,
    (5,): 4,
    (6,): 4,
    (7,): 4,
    (8,): 4,
    (9,): 4,
    (10,): 2 * 4,
    (11,): 2 * 4,
}
top_k = 1
use_residual = True


def run_test(rank, port):
    if rank != 0:
        return
    # 1. Configure for Parallelization
    print(f"# 1. Configure for Parallelization", flush=True)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 2. Set Parallel Context
    print("# 2. Set Parallel Context", flush=True)
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=world_size // expert_parallel_size,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=expert_parallel_size,
    )

    # 3. Create Tokenizer
    print("# 3. Create Tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Create Model to expert-parallelize
    print("# 4. Create Model to expert-parallelize", flush=True)
    model_ep_load = GPT2LMHeadModel(GPT2Config.from_pretrained("./gpt2_ep"))

    # 5. Wrap Model
    print("# 5. Wrap Model", flush=True)
    wrapper_ep = ExpertParallel(
        model_ep_load,
        parallel_context,
        num_dec_experts=num_experts,
        top_k=1,
        use_residual=use_residual,
        extra_states_path="./gpt2_ep/",
    ).to(rank)

    oslo.ready(wrapper_ep, parallel_context)

    # 6. Prepare Optimizer
    print("# 6. Prepare Optimizer", flush=True)
    optimizer = Adam(wrapper_ep.parameters(), lr=3e-5)

    # 7. Prepare Dataset
    print("# 7. Prepare Dataset", flush=True)
    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:500]]
    dataloader = DataLoader(datasets, batch_size=4)

    # 8. Forward Propagation
    print("# 8. Forward Propagation", flush=True)
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(rank)
        loss = wrapper_ep(**inputs, labels=inputs["input_ids"]).loss
        if rank == 0:
            print(f"Rank #{rank} Iteration #{i} loss : {loss}")

        loss.backward()
        optimizer.step()


def test_expert_parallel_block():
    run_parallel = partial(run_test, port=29500)
    mp.spawn(run_parallel, nprocs=1)

    # run_deparallel = partial(run_test, port=29500, mode="deparallelize")
    # mp.spawn(run_deparallel, nprocs=world_size)


if __name__ == "__main__":
    # Set Random Seed for Reproducibility
    # fix_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
