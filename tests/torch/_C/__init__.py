from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.distributed import rpc
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    set_seed,
)

import oslo
from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.pipeline_parallel._buffers import _MODULE_DEVICE_LOCATIONS


# set matploblib
matplotlib.use("Agg")
torch.autograd.set_detect_anomaly(True)
set_seed(42)

# create mpu
parallel_context = ParallelContext.from_torch(pipeline_parallel_size=2)
current_device = torch.cuda.current_device()

# create model
model_name = "t5-small"
model = T5ForConditionalGeneration(T5Config.from_pretrained(model_name))

# for deterministic.
for n, m in model.named_modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

# create non-pp model to compare
model_no_pp = deepcopy(model)
model_no_pp.cuda()

# create pp wrapper
num_micro_batches = 8
wrapper_pp = PipelineParallel(
    model,
    parallel_context=parallel_context,
    memory_computation_balance=1.0,
    num_micro_batches=num_micro_batches,
)

# allocate model to gpu
oslo.ready(wrapper_pp, parallel_context)


# create optimizer
optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-2)
optimizer_no_pp = Adam(model_no_pp.parameters(), lr=3e-2)


if torch.distributed.get_rank() == 1:
    for k, v in _MODULE_DEVICE_LOCATIONS.items():
        print(f"{k}: cuda:{v}")


def run():
    batch_size = 2 * num_micro_batches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:5000]]
    dataloader = DataLoader(datasets, batch_size=batch_size)
    pp_losses = []
    no_pp_losses = []

    for i, data in enumerate(dataloader):
        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        optimizer_pp.zero_grad(set_to_none=True)
        optimizer_no_pp.zero_grad(set_to_none=True)

        cum_loss_pp = torch.zeros(1)
        for out_pp in wrapper_pp(**inputs, labels=inputs["input_ids"]):
            loss_pp = out_pp.loss
            loss_pp.backward()

            cum_loss_pp += loss_pp.detach().item()

        out_no_pp = model_no_pp(**inputs, labels=inputs["input_ids"])
        loss_no_pp = out_no_pp.loss
        loss_no_pp.backward()

        if dist.get_rank() == 0:
            print(f"{dist.get_rank()}, {cum_loss_pp}, {loss_no_pp}")

        optimizer_pp.step()
        optimizer_no_pp.step()

        pp_losses.append(cum_loss_pp.item())
        no_pp_losses.append(loss_no_pp.item())

    if dist.get_rank() == 0:
        plt.figure(figsize=(32, 8))
        plt.plot(pp_losses, label="PP")
        plt.plot(no_pp_losses, label="no PP")
        plt.legend()
        plt.title(f"{model_name}")
        plt.savefig(f"{model_name} pp_vs_no_pp.png")

    torch.distributed.rpc.shutdown()


if __name__ == "__main__":
    run()
