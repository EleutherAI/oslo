import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

import oslo
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel

parser = argparse.ArgumentParser()
parser.add_argument("--memory_priority", action="store_true", default=False)
args = parser.parse_args()


def seed_all(seed: int = 1930):

    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_all(seed=1994)


def latency_trace(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start

    return wrapper


@latency_trace
def fw(func, *args, **kwargs):
    return func(*args, **kwargs).loss


@latency_trace
def bw(tensors):
    return tensors.backward()


tp_size = 8
tp_depth = 1

model_name = "jason9693/soongsil-bert-base"
mkwargs = {}
dataset_name = "squad_kor_v1"

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_3D,
    tensor_parallel_depth=tp_depth,
)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(model_name, **mkwargs)
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_tp = AutoModelForCausalLM.from_config(
    AutoConfig.from_pretrained(model_name)
).cuda()
model_tp = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
wrapper_tp = TensorParallel(
    model_tp, parallel_context, memory_priority=args.memory_priority
)
oslo.ready(wrapper_tp, parallel_context)

if dist.get_rank() == 0:
    print(wrapper_tp)

# 옵티마이저 생성
optimizer_tp = Adam(wrapper_tp.parameters(), lr=3e-5)
optimizer_no_tp = Adam(model_no_tp.parameters(), lr=3e-5)

# 데이터셋 생성
batch_size = 16
datasets = load_dataset(dataset_name).data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"{model_name}_tp3d_bs{batch_size}")
    cur = time.time()

# 저장
wrapper_tp.save_pretrained(save_directory="test/", merge_checkpoints=True)

# 모니터링 생성 대기
dist.barrier()

# 로드
model_gathered = AutoModelForCausalLM.from_pretrained("test/").cuda()
optimizer_gathered = Adam(model_gathered.parameters(), lr=3e-5)

dist.barrier()

# 학습 시작
for data in dataloader:
    optimizer_tp.zero_grad()
    optimizer_no_tp.zero_grad()
    optimizer_gathered.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_no_tp, notp_fw_time = fw(model_no_tp, **inputs, labels=inputs["input_ids"])
    loss_tp, tp_fw_time = fw(wrapper_tp, **inputs, labels=inputs["input_ids"])
    loss_gathered, gathered_fw_time = fw(
        model_gathered, **inputs, labels=inputs["input_ids"]
    )

    if dist.get_rank() == 0:
        print(f"TP:{loss_tp}, NOTP:{loss_no_tp}, GATHRED:{loss_gathered}")
        wandb.log({"tp": loss_tp, "notp": loss_no_tp, "GATHRED": loss_gathered})

    _, notp_bw_time = bw(loss_no_tp)
    _, tp_bw_time = bw(loss_tp)
    _, gathered_bw_time = bw(loss_gathered)

    optimizer_tp.step()
    optimizer_no_tp.step()
    optimizer_gathered.step()

    if dist.get_rank() == 0:
        wandb.log(
            {
                "tp.forward.time:": tp_fw_time,
                "tp.backward.time:": tp_bw_time,
                "notp.forward.time:": notp_fw_time,
                "notp.backward.time:": notp_bw_time,
                "gathered.forward.time:": gathered_fw_time,
                "gathered.backward.time:": gathered_bw_time,
            }
        )

dist.barrier()
