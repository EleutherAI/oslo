import os.path
import time
import types
import glob
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.distributed import rpc
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    set_seed,
)

from oslo.torch.distributed import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import parallelize
from oslo.transformers.constants import BATCH_DIMENSIONS_PP


set_seed(42)

data_parallel_size = 1
parallel_context = ParallelContext.from_torch(
    data_parallel_size=data_parallel_size,
    pipeline_parallel_size=4,
    tensor_parallel_size=1,
)

current_device = torch.cuda.current_device()
num_micro_batches = 1

model_name = "gpt2"
config = GPT2Config.from_pretrained(model_name)

model = GPT2LMHeadModel(config)

# experimental
model.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

for n, m in model.named_modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

model_no_pp = deepcopy(model)
model_no_pp.cuda()


wrapper_pp = PipelineParallel(
    model,
    parallel_context=parallel_context,
    memory_computation_balance=1.0,
    num_micro_batches=num_micro_batches,
)

wrapper_pp.train()

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-4)
optimizer_no_pp = Adam(model_no_pp.parameters(), lr=3e-4)

parallelize(wrapper_pp, parallel_context)

save_dir = "tmp_0"
if save_dir is not None:

    def save_output_hook(name, is_parallel):

        def hook(module, inp, outp):

            if name == "":
                return

            if torch.is_tensor(outp):
                outp_cpu = outp.cpu()
            else:
                if isinstance(outp, tuple):
                    outp_cpu = []
                    for x in outp:
                        if torch.is_tensor(x):
                            outp_cpu.append(x.cpu())
                        else:
                            outp_cpu.append(x)
                    outp_cpu = tuple(outp_cpu)
                else:
                    outp_cpu = outp

            if is_parallel:
                if isinstance(outp, types.GeneratorType):
                    return
                torch.cuda.synchronize()
                torch.save(
                    outp_cpu,
                    f"{save_dir}/output_{name}_pp_tp_{torch.distributed.get_rank()}.pkl",
                )
            else:
                torch.cuda.synchronize()
                torch.save(outp_cpu, f"{save_dir}/output_{name}_no_pp_tp.pkl")

        return hook

    for name, m in wrapper_pp.named_modules():
        m.register_forward_hook(save_output_hook(name, True))

    for name, m in model_no_pp.named_modules():
        m.register_forward_hook(save_output_hook(name, False))


def check_lm_head():
    grad_sent_path = "sent_grad.pkl"
    grad_sent_data = torch.load(grad_sent_path, map_location="cpu")

    grad_received_path = "saved_grad.pkl"
    grad_received_data = torch.load(grad_received_path, map_location="cpu")

    if not torch.allclose(grad_sent_data, grad_received_data):
        print(f" lm.head grad  >>> diff: {torch.sum(torch.abs(grad_sent_data - grad_received_data))}")


def check_gradient():
    file_names = os.listdir(f"{save_dir}/")
    no_pp_names = sorted([fn for fn in file_names if "no_pp" in fn and "grad_" in fn])

    diff_cnt = 0
    diff_names = []
    same_names = []
    for no_pp_name in no_pp_names:
        pp_tp_name_template = no_pp_name.split("no_pp")[0]
        pp_tp_name_template = pp_tp_name_template + "pp_tp_*.pkl"
        pp_tp_paths = glob.glob(os.path.join(f"{save_dir}", pp_tp_name_template))
        pp_tp_paths = sorted(pp_tp_paths)

        no_pp_path = os.path.join(f"{save_dir}", no_pp_name)
        no_pp_data = torch.load(no_pp_path, map_location="cpu")

        pp_tp_data = []
        for path in pp_tp_paths:
            data = torch.load(path, map_location="cpu")
            pp_tp_data.append(data)

        pp_tp_data = pp_tp_data[0]

        if not torch.is_tensor(no_pp_data):
            print(type(no_pp_data), no_pp_name)
            continue

        if not torch.allclose(pp_tp_data, no_pp_data):
            print(f" {no_pp_name}  >>> diff: {torch.sum(torch.abs(pp_tp_data - no_pp_data))}")

            diff_cnt += 1
            diff_names.append(no_pp_name)

        else:
            same_names.append(no_pp_name)


def check_output():
    file_names = os.listdir(f"{save_dir}/")
    no_pp_names = sorted([fn for fn in file_names if "no_pp" in fn and "output_" in fn])

    diff_cnt = 0
    diff_names = []
    same_names = []

    for no_pp_name in no_pp_names:
        pp_tp_name_template = no_pp_name.split("no_pp")[0]
        pp_tp_name_template = pp_tp_name_template + "pp_tp_*.pkl"
        pp_tp_paths = glob.glob(os.path.join(f"{save_dir}", pp_tp_name_template))
        pp_tp_paths = sorted(pp_tp_paths)

        no_pp_path = os.path.join(f"{save_dir}", no_pp_name)
        no_pp_data = torch.load(no_pp_path, map_location="cpu")

        pp_tp_data = []
        for path in pp_tp_paths:
            data = torch.load(path, map_location="cpu")
            pp_tp_data.append(data)

        pp_tp_data = pp_tp_data[0]

        do_print = False
        if not torch.is_tensor(no_pp_data):
            if isinstance(no_pp_data, tuple):
                no_pp_data = [no_pp_data[0]]
                pp_tp_data = [pp_tp_data[0]]

            else:
                # model output; transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions
                # print(type(no_pp_data), type(pp_tp_data))
                no_pp_data = [no_pp_data.last_hidden_state]
                pp_tp_data = [pp_tp_data.last_hidden_state]
                do_print = True

        else:
            no_pp_data = [no_pp_data]
            pp_tp_data = [pp_tp_data]

        for x, y in zip(pp_tp_data, no_pp_data):
            if not torch.allclose(x, y) or do_print:
                print(f" {no_pp_name}  >>> diff: {torch.sum(torch.abs(x - y))}")
                diff_cnt += 1
                diff_names.append(no_pp_name)

            else:
                same_names.append(no_pp_name)


def run():
    batch_size = 128 * num_micro_batches
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:8192]]
    dataloader = DataLoader(datasets, batch_size=batch_size)

    pp_losses = []
    no_pp_losses = []

    step_count = 0
    with torch.enable_grad():
        for i, data in enumerate(dataloader):
            global save_dir
            save_dir = f"tmp_{i}"
            os.makedirs(save_dir, exist_ok=True)

            inputs = tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to("cuda")

            inputs["input_ids"][inputs["input_ids"] == tokenizer.pad_token] = -100

            if data_parallel_size > 1:
                dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
                new_inputs = dict()
                for key, value in inputs.items():  # assumes HF
                    new_inputs[key] = value.chunk(data_parallel_size)[dp_rank]
                pp_inputs = new_inputs
            else:
                pp_inputs = inputs

            torch.distributed.barrier()

            optimizer_pp.zero_grad(set_to_none=True)
            optimizer_no_pp.zero_grad(set_to_none=True)

            cum_loss_pp = torch.zeros(1).cuda()
            partial_loss_pp = [None for _ in range(num_micro_batches)]

            for _, (ind, out_pp) in enumerate(
                wrapper_pp(**pp_inputs, labels=pp_inputs["input_ids"])
            ):

                if out_pp is not None:
                    loss_pp = out_pp.loss
                    # loss is scaled in PP wrapper
                    loss_pp.backward()

                    _l = loss_pp.detach().item()
                    cum_loss_pp += _l
                    partial_loss_pp[ind] = _l

                else:
                    pass

            cum_loss_no_pp = torch.zeros(1).cuda()
            partial_loss_no_pp = [None for _ in range(num_micro_batches)]

            new_inputs = [dict() for _ in range(num_micro_batches)]
            for key, value in inputs.items():
                if key in BATCH_DIMENSIONS_PP:
                    # splittable
                    value = value.chunk(
                        num_micro_batches,
                        dim=BATCH_DIMENSIONS_PP[key],
                    )
                    for ind, v_chunk in enumerate(value):
                        new_inputs[ind][key] = v_chunk
                else:
                    # not splittable
                    for ind in range(num_micro_batches):
                        new_inputs[ind][key] = value

            for ind, inputs in enumerate(new_inputs):
                out_no_pp = model_no_pp(**inputs, labels=inputs["input_ids"])
                loss_no_pp = out_no_pp.loss / num_micro_batches
                loss_no_pp.backward()

                _l = loss_no_pp.detach().item()
                cum_loss_no_pp += _l
                partial_loss_no_pp[ind] = _l

            torch.distributed.barrier()

            if dist.get_rank() == 0:
                print(f"{dist.get_rank()}, {cum_loss_pp}, {cum_loss_no_pp}")

            torch.distributed.barrier()

            if save_dir is not None:
                for name, param in wrapper_pp.named_parameters():
                    if param.grad is not None:
                        torch.save(
                            param.grad,
                            f"{save_dir}/grad_{name}_pp_tp_{torch.distributed.get_rank()}.pkl",
                        )

                if dist.get_rank() == 0:
                    for name, param in model_no_pp.named_parameters():
                        torch.save(param.grad, f"{save_dir}/grad_{name}_no_pp.pkl")

                torch.distributed.barrier()

                if dist.get_rank() == 0:
                    print(save_dir)

                    print(f"Start to check lm.head:")
                    check_lm_head()

                    # check
                    print(f"Start to check output:")
                    check_output()
                    print(f"Output check done.")

                    print(f"Start to check gradient:")
                    check_gradient()
                    print(f"Gradient check done.")

            torch.distributed.barrier()
            optimizer_pp.step()

            torch.distributed.barrier()

            optimizer_no_pp.step()
            torch.distributed.barrier()

            step_count += 1

            pp_losses.append(cum_loss_pp.item())
            no_pp_losses.append(cum_loss_no_pp.item())

    torch.distributed.rpc.shutdown()

    if dist.get_rank() == 0:
        plt.figure(figsize=(32, 8))
        plt.plot(pp_losses, label="PP")
        plt.plot(no_pp_losses, label="no PP")
        plt.legend()
        plt.title(f"{model_name}")
        plt.savefig(f"{model_name} pp_vs_no_pp rank4.png")


if __name__ == "__main__":
    run()
