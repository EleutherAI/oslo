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

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel.expert_parallel.expert_parallel import _ExpertParallel
from oslo.torch.nn.parallel.data_parallel.data_parallel import DistributedDataParallel


from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

import wandb

os.environ["TRANSFORMER_CACHE"] = "/fsx/seokung/.cache"


torch.set_printoptions(threshold=10_000)

world_size = 4
expert_parallel_size = 2
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


def run_test(rank, port):
    # 1. Configure for Parallelization
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    if rank == 0:
        wandb.init(project="oslo", name="ep")

    # 2. Set Parallel Context
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=world_size // expert_parallel_size,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=expert_parallel_size,
    )

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
    wrapper_ep = DistributedDataParallel(wrapper_ep, parallel_context=parallel_context)

    # 6. Prepare Optimizer
    optimizer = Adam(wrapper_ep.parameters(), lr=3e-5)

    # 7. Prepare Dataset
    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:500]]
    dataloader = DataLoader(datasets, batch_size=4)

    # 8. Check Parameter
    # for param_name, module in wrapper_ep.named_parameters():
    #    print(
    #         f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
    #    )
    #    print(f"Worker #{rank} - param  : {module}")
    # if wrapper_ep.expert_parallel_mapping.is_front_parallel(
    #    wrapper_ep.model, param_name
    # ) or wrapper_ep.expert_parallel_mapping.is_behind_parallel(
    #    wrapper_ep.model, param_name
    # ):
    #    print(
    #        f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
    #    )
    #    print(f"Worker #{rank} - param  : {module}")

    # 9. Forward Propagation
    for i, data in enumerate(dataloader):
        start = time.time()
        optimizer.zero_grad()

        inputs = tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(rank)
        loss = wrapper_ep(**inputs, labels=inputs["input_ids"]).loss
        if rank == 0:
            print(f"Rank #{rank} Iteration #{i} loss : {loss}")
            wandb.log({"iter": i, "ep_loss": loss, "iter_time": time.time() - start})

        loss.backward()
        optimizer.step()


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # Set Random Seed for Reproducibility
    # fix_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
