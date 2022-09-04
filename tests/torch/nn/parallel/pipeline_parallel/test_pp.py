from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import rpc
from torch.optim import Adam
from torch.utils.data import DataLoader

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, set_seed

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
torch.autograd.set_detect_anomaly(True)
set_seed(42)

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
)

current_device = torch.cuda.current_device()

model_name = "gpt2"
num_micro_batches = 8

config = GPT2Config.from_pretrained(model_name)

model = GPT2LMHeadModel(config)

for n, m in model.named_modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.

model_no_pp = deepcopy(model)
model_no_pp.cuda()

wrapper_pp = PipelineParallel(
    model,
    parallel_context=parallel_context,
    memory_computation_balance=1.0,
    num_micro_batches=num_micro_batches,
)

wrapper_pp.train()

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-5)
optimizer_no_pp = Adam(model_no_pp.parameters(), lr=3e-5)

allocate_params(wrapper_pp, parallel_context)


def run():
    batch_size = 8 * num_micro_batches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:5000]]
    dataloader = DataLoader(datasets, batch_size=batch_size)

    pp_losses = []
    no_pp_losses = []

    with torch.enable_grad():
    # with torch.no_grad():
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
            for ind, out_pp in enumerate(wrapper_pp(**inputs, labels=inputs["input_ids"])):
                loss_pp = out_pp.loss
                loss_pp = loss_pp / num_micro_batches

                if dist.get_rank() == 0:
                    loss_pp.backward()

                print(f'{ind=}')
                cum_loss_pp += loss_pp.detach().item()

            out_no_pp = model_no_pp(**inputs, labels=inputs["input_ids"])
            loss_no_pp = out_no_pp.loss
            loss_no_pp.backward()

            print(f'{dist.get_rank()=}, {cum_loss_pp=}, {loss_no_pp=}')

            optimizer_pp.step()
            optimizer_no_pp.step()

            pp_losses.append(cum_loss_pp.item())
            no_pp_losses.append(loss_no_pp.item())

    torch.distributed.rpc.shutdown()

    if dist.get_rank() == 0:
        plt.figure(figsize=(32, 8))
        plt.plot(pp_losses, label='PP')
        plt.plot(no_pp_losses, label='no PP')
        plt.legend()
        plt.title(f'{model_name}')
        plt.savefig(f'{model_name} pp_vs_no_pp.png')


run()

