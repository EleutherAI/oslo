import os
import random
import time
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.expert_parallel.expert_parallel import _ExpertParallel
from oslo.torch.nn.parallel.data_parallel.data_parallel import DistributedDataParallel


from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

import wandb

os.environ["TRANSFORMER_CACHE"] = "/fsx/seokung/.cache"

save_path = "./gpt2_deparallelize.ckt"
# torch.set_printoptions(threshold=10_000)

world_size = 8
expert_parallel_size = 4
num_experts = {
    (0,): expert_parallel_size,
    (1,): expert_parallel_size,
    (2,): expert_parallel_size,
    (3,): expert_parallel_size,
    (4,): expert_parallel_size,
    (5,): expert_parallel_size,
    (6,): expert_parallel_size,
    (7,): expert_parallel_size,
    (8,): expert_parallel_size,
    (9,): expert_parallel_size,
    (10,): 2 * expert_parallel_size,
    (11,): 2 * expert_parallel_size,
}
top_k = 1
use_residual = True


def run_test(rank, port, mode):
    # 1. Configure for Parallelization
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # if rank == 0 and mode != 'save':
    #    wandb.init(project="oslo", name="ep")

    # 2. Set Parallel Context
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=world_size // expert_parallel_size,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=expert_parallel_size,
    )
    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    ep_rank = parallel_context.get_local_rank(ParallelMode.EXPERT)

    # 3. Create Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Create Model to expert-parallelize
    model_ep = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

    # 5. Wrap Model
    wrapper_ep = _ExpertParallel(
        model_ep,
        parallel_context,
        num_dec_experts=num_experts,
        top_k=1,
        use_residual=use_residual,
    ).to(rank)

    # 6. Deparallelize
    if mode == "deparallelize":
        wrapper_ep.deparallelize()
        if dp_rank != 0 or ep_rank != 0:
            return
    elif mode != "parallelize":
        raise NotImplementedError(f"{mode} is not supported for test deparallization")

    # 7. Check Parameter
    for param_name, module in wrapper_ep.named_parameters():
        if "expert" not in param_name:
            continue
        print(
            f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
        )
        print(f"Worker #{rank} - param  : {module}")


def test_expert_parallel_block():
    # run_parallel = partial(run_test, port=29500, mode='parallelize')
    # mp.spawn(run_parallel, nprocs=world_size)

    run_deparallel = partial(run_test, port=29500, mode="deparallelize")
    mp.spawn(run_deparallel, nprocs=world_size)


if __name__ == "__main__":
    # Set Random Seed for Reproducibility
    # fix_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
