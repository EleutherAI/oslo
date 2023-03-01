import os
import random
import time
from functools import partial

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from torch.optim import Adam
from torch.utils.data import DataLoader

import deepspeed
from deepspeed.utils import groups
from deepspeed.moe.layer import MoE

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

torch.set_printoptions(threshold=10_000)

world_size = 4

top_k = 1
use_residual = True

hidden_size = 768

ep_size = world_size // 2
num_experts = {
    (0,): ep_size,
    (1,): ep_size,
    (2,): ep_size,
    (3,): ep_size,
    (4,): ep_size,
    (5,): ep_size,
    (6,): ep_size,
    (7,): ep_size,
    (8,): ep_size,
    (9,): ep_size,
    (10,): 2 * ep_size,
    (11,): 2 * ep_size,
}


def get_layer_id(param_name):
    param_name = param_name.split(".")
    for cur in param_name:
        if cur.isdigit():
            return int(cur)
    return


def run_test(rank, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 1. Init process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 2. Create Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Create Model to expert-parallelize
    model_ep = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

    # 4. Check Parameter
    named_modules = [
        (module_name, module) for module_name, module in model_ep.named_modules()
    ]
    for module_name, module in named_modules:
        if "mlp" == module_name.split(".")[-1]:
            layer_id = get_layer_id(module_name)
            # print(
            #        f"Worker #{rank} - module_name : {module_name}, layer_id : {layer_id}"
            # )
            moe = MoE(
                hidden_size=hidden_size,
                expert=module,
                num_experts=num_experts[(layer_id,)],
                ep_size=ep_size,
                k=top_k,
                use_residual=use_residual,
                use_tutel=False,
            )
            # module.__class__ = MoE
            setattr(model_ep, module_name, moe)
            # print(moe)
    model_ep.to(rank)

    # 5. Create Optimizer
    optimizer = Adam(model_ep.parameters(), lr=3e-5)

    # 6. Create Datasets
    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:500]]
    dataloader = DataLoader(datasets, batch_size=4)

    # 7. Train
    for i, data in enumerate(dataloader):
        start = time.time()
        optimizer.zero_grad()

        inputs = tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(rank)
        loss = model_ep(**inputs, labels=inputs["input_ids"]).loss
        if rank == 0:
            print(f"Rank #{rank} Iteration #{i} loss : {loss}")
        loss.backward()
        optimizer.step()


def test_deepspeed_moe():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_deepspeed_moe()
