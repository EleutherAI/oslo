import random
import torch.distributed as dist
import wandb
import numpy as np
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext
import oslo
import argparse

# parallel context 생성
from oslo.torch.nn.parallel.data_parallel.data_parallel import DataParallel


def get_parallel_context(configs):
    # parallel context 생성
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=configs["data_parallel_size"],
        pipeline_parallel_size=configs["pipeline_parallel_size"],
        sequence_parallel_size=configs["sequence_parallel_size"],
    )

    return parallel_context


def get_test_train_dataloader(
    training_type, test_dataset="squad", batch_size=2, dataset_length=4000
):

    # 데이터셋 생성
    datasets = load_dataset(test_dataset).data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:dataset_length]]

    world_size = dist.get_world_size()

    if training_type == "ddp":
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets, num_replicas=world_size, shuffle=False
        )
        batch_size = int(batch_size // world_size)
        dataloader = DataLoader(
            datasets, batch_size=batch_size, sampler=train_sampler, shuffle=False
        )
    else:
        dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False)

    return dataloader


def train(training_type, dataloader, model, optimizer, tokenizer, configs):
    # 학습 시작
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        loss = model(**inputs, labels=inputs["input_ids"]).loss

        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:
            wandb.log({training_type + " Loss": loss}, step=i, commit=False)
            # loss_record.append(loss)


def run_dp_gpt2_test(parallel_context, configs):

    # 토크나이저 생성
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 생성 및 병렬화 수행
    model_no_ddp = GPT2LMHeadModel(
        GPT2Config.from_pretrained(configs["model_name"])
    ).cuda()

    model_ddp = GPT2LMHeadModel(GPT2Config.from_pretrained(configs["model_name"]))

    # 옵티마이저 생성
    optimizer_no_ddp = Adam(model_no_ddp.parameters(), lr=3e-5)
    optimizer_ddp = Adam(model_ddp.parameters(), lr=3e-5)

    model_ddp, optimizer_ddp = DataParallel(
        module=model_ddp,
        optimizer=optimizer_ddp,
        parallel_context=parallel_context,
        zero_stage=1,
    )
    oslo.ready(model_ddp, parallel_context)

    # TODO replace hardcoded number with
    dataloader_no_ddp = get_test_train_dataloader(
        "no_ddp", configs["dataset_name"], 4, 500
    )
    # batch size = 2 per a model
    dataloader_ddp = get_test_train_dataloader("ddp", configs["dataset_name"], 4, 500)

    # 모니터링 생성
    if dist.get_rank() == 0:
        wandb.init(project="oslo", group="fsdp7")

    # 학습 시작
    train(
        "no_ddp", dataloader_no_ddp, model_no_ddp, optimizer_no_ddp, tokenizer, configs
    )
    train("ddp", dataloader_ddp, model_ddp, optimizer_ddp, tokenizer, configs)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_size", default=2)
    parser.add_argument("--pipeline_parallel_size", default=1)
    parser.add_argument("--sequence_parallel_size", default=1)
    parser.add_argument("--dataset_name", default="squad")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--epochs", default=1)

    args, remaining_argv = parser.parse_known_args()

    configs = {}

    for k in args.__dict__:
        configs[k] = args.__dict__[k]

    print("configs info :", configs)

    return configs


def run(configs):

    parallel_context = get_parallel_context(configs)

    if configs["model_name"] == "gpt2":
        run_dp_gpt2_test(parallel_context, configs)


if __name__ == "__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # transformers.set_seed(seed)

    # TODO
    # handling config (json or yml file)
    configs = args()
    # execute cofnigured test
    # 1. gpt2 with squad
    run(configs)
