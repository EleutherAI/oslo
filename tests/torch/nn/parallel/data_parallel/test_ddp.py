import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext, ParallelMode
import oslo
import argparse
import numpy as np
import random

# parallel context 생성
from oslo.torch.nn.parallel.data_parallel._ddp.distributed_data_parallel import (
    DistributedDataParallel as DistributedDataParallelv1,
)

from oslo.torch.nn.parallel.data_parallel._coloddp.distributed_data_parallel import (
    DistributedDataParallel,
)


def get_parallel_context(configs):
    # parallel context 생성
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=configs["data_parallel_size"],
        pipeline_parallel_size=configs["pipeline_parallel_size"],
        sequence_parallel_size=configs["sequence_parallel_size"],
    )

    return parallel_context


def get_test_train_dataloader(
    training_type, test_dataset="squad", batch_size=2, dataset_length=500
):

    # 데이터셋 생성

    datasets = load_dataset(test_dataset).data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:dataset_length]]

    world_size = dist.get_world_size()

    if training_type == "ddp":
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets,
            num_replicas=world_size,
            shuffle=False,
        )
        batch_size = int(batch_size // world_size)
        dataloader = DataLoader(
            datasets, batch_size=batch_size, sampler=train_sampler, shuffle=False
        )
    else:
        dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False)

    return dataloader


class GPTLMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


def train(training_type, dataloader, model, optimizer, tokenizer):
    # 학습 시작
    model.train()

    for data in dataloader:
        optimizer.zero_grad()

        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        loss = model(**inputs, labels=inputs["input_ids"]).loss

        if dist.get_rank() == 0:
            wandb.log({training_type + " Loss": loss})

        loss.backward()
        optimizer.step()


def run_dp_gpt2_test(parallel_context, configs):

    # 토크나이저 생성
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 생성 및 병렬화 수행
    model_no_ddp = GPT2LMHeadModel(
        GPT2Config.from_pretrained(configs["model_name"])
    ).cuda()

    model_ddp = GPT2LMHeadModel(GPT2Config.from_pretrained(configs["model_name"]))
    model_ddp = DistributedDataParallelv1(
        model_ddp,
        parallel_context=parallel_context,
    )
    oslo.ready(model_ddp, parallel_context)

    datasets = load_dataset(configs["dataset_name"]).data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:500]]

    # 옵티마이저 생성
    optimizer_no_ddp = Adam(model_no_ddp.parameters(), lr=3e-5)
    optimizer_ddp = Adam(model_ddp.parameters(), lr=3e-5)

    # TODO replace hardcoded number with
    # batch size = 2 per a model
    dataloader_no_ddp = get_test_train_dataloader(
        "no_ddp", configs["dataset_name"], 4, 500
    )
    dataloader_ddp = get_test_train_dataloader("ddp", configs["dataset_name"], 4, 500)

    # 모니터링 생성
    if dist.get_rank() == 0:
        wandb.init(project="oslo", group="ddp")

    # 학습 시작
    train("no_ddp", dataloader_no_ddp, model_no_ddp, optimizer_no_ddp, tokenizer)
    train("ddp", dataloader_ddp, model_ddp, optimizer_ddp, tokenizer)


def run_coloddp_gpt2_test(parallel_context, configs):

    # 토크나이저 생성
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # 모델 생성 및 병렬화 수행
    model_no_ddp = GPT2LMHeadModel(
        GPT2Config.from_pretrained(configs["model_name"])
    ).cuda()

    model_ddp = GPT2LMHeadModel(GPT2Config.from_pretrained(configs["model_name"]))

    model_ddp = DistributedDataParallel(
        model_ddp,
        parallel_context=parallel_context,
    )
    oslo.ready(model_ddp, parallel_context)

    datasets = load_dataset(configs["dataset_name"]).data["train"]["context"]
    datasets = [str(sample) for sample in datasets[:500]]

    # 옵티마이저 생성
    optimizer_no_ddp = Adam(model_no_ddp.parameters(), lr=3e-5)
    optimizer_ddp = Adam(model_ddp.parameters(), lr=3e-5)

    # TODO replace hardcoded number with
    # batch size = 2 per a model
    dataloader_no_ddp = get_test_train_dataloader(
        "no_ddp", configs["dataset_name"], 4, 500
    )

    dataloader_ddp = get_test_train_dataloader("ddp", configs["dataset_name"], 4, 500)

    # 모니터링 생성
    if dist.get_rank() == 0:
        wandb.init(project="oslo", group="ddp")

    # # 학습 시작
    train("no_ddp", dataloader_no_ddp, model_no_ddp, optimizer_no_ddp, tokenizer)

    # coloddp case
    criterion = GPTLMLoss()

    # 학습 시작
    model_ddp.train()

    for data in dataloader_ddp:
        optimizer_ddp.zero_grad()

        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        outputs = model_ddp(**inputs)[0]
        loss = criterion(outputs, inputs["input_ids"])

        if dist.get_rank() == 0:
            wandb.log({"coloddp" + " Loss": loss})

        model_ddp.backward(loss)
        optimizer_ddp.step()


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parallel_size", default=2)
    parser.add_argument("--pipeline_parallel_size", default=1)
    parser.add_argument("--sequence_parallel_size", default=1)
    parser.add_argument("--dataset_name", default="squad")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--ddp_mode", default="ddp")

    args, remaining_argv = parser.parse_known_args()

    configs = {}

    for k in args.__dict__:
        configs[k] = args.__dict__[k]

    print("configs info :", configs)

    return configs


def run(configs):

    parallel_context = get_parallel_context(configs)

    if configs["model_name"] == "gpt2":
        if configs["ddp_mode"] == "coloddp":
            run_coloddp_gpt2_test(parallel_context, configs)
        else:
            print("pytorch native torch ddp")
            run_dp_gpt2_test(parallel_context, configs)


if __name__ == "__main__":
    # TODO
    # handling config (json or yml file)

    # fix random variables
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    configs = args()
    # execute cofnigured test
    run(configs)
