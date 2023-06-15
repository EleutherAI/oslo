import time
import torch
import numpy as np
import random
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from datasets import Dataset as Dt
from datasets import DatasetDict, concatenate_datasets
import os

from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)
from tests_deprecated.transformers.tasks.test_data_base import TestDataBinarization
from transformers import (
    BartTokenizerFast,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    get_scheduler,
    BartModel,
)

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


def get_wiki_data():
    with open("./wiki_output_file.txt", encoding="utf8") as f:
        wiki_texts = [line.replace("\n", "") for line in f.readlines()]
    dataset = DatasetDict({"train": Dt.from_dict({"text": wiki_texts})})
    return dataset


def get_cnn_data():
    dataset = load_dataset(
        # data_args.dataset_name, ############ !!!!!!!!!!!!!!!!!!
        "cnn_dailymail",
        "3.0.0",
        use_auth_token=None,
        # split = "train+validation+test"
        # use_auth_token=True if model_args.use_auth_token else None,
    )

    dataset = dataset.rename_column("article", "text")
    dataset = dataset.remove_columns(["highlights", "id"])
    # # reduce dataset for time
    dataset["train"] = Dt.from_dict(dataset["train"][:4523])
    dataset["validation"] = Dt.from_dict(dataset["validation"][:555])
    dataset["test"] = Dt.from_dict(dataset["test"][:555])

    dataset = DatasetDict(
        {
            "train": concatenate_datasets(
                [dataset["train"], dataset["validation"], dataset["test"]]
            )
        }
    )

    return dataset


if "__main__" == __name__:

    # GPU settings
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    # dist.init_process_group(bacskend="nccl",world_size=world_size, rank=rank)
    # torch.cuda.set_device(rank)
    # device = torch.cuda.current_device()
    print(f"rank = {rank} world_size = {world_size} device = {device}")
    dataset_name = "wiki"  # cnn or wiki
    dataset = get_cnn_data() if dataset_name == "cnn" else get_wiki_data()

    model_nm = "facebook/bart-base"
    max_seq_length = 512  # user setting

    # Model Config Setting
    config = BartConfig(
        vocab_size=50265,
        max_position_embeddings=1024,
        d_model=1024,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_start_token_id=2,
        dropout=0.1,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        num_hidden_layers=6,
    )
    # seed setting for reproducible
    seed_num = 1993
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)  # if use multi-GPU

    tokenizer = BartTokenizerFast.from_pretrained(model_nm)
    processor = ProcessorForBartPretraining(tokenizer, max_seq_length=max_seq_length)

    data_collator = DataCollatorForBartPretraining(
        processor,
        label_pad_token_id=-100,
        mask_ratio=0.3,  # for text infilling
        poisson_lambda=3.0,  # for text infilling
        permute_sentence_ratio=1.0,  # for sentence shuffle
        # permute_sentence_ratio = 0.0
    )
    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        tokenizer = data_collator.tokenizer
        print("pad_token is set.")

    processed_dataset = processor(dataset)
    # Code to save and load the prepared data to disk if the training data preparation time is long.
    # code save
    # processed_dataset.save_to_disk(f'/home/bsj/.cache/huggingface/datasets/{dataset_name}_processed')
    # code load
    # processed_dataset = dataset.load_from_disk(f'/home/bsj/.cache/huggingface/datasets/{dataset_name}_processed')

    # for Distributed Data Parallel
    per_device_train_batch_size = 36
    strategy = "ddp"  # ddp or None
    dataloader_worker = 3 * 4  # user setting
    train_sampler = (
        DistributedSampler(
            processed_dataset["train"], num_replicas=world_size, rank=rank, shuffle=True
        )
        if strategy == "ddp"
        else None
    )

    train_dataloader = DataLoader(
        dataset=processed_dataset["train"],
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        sampler=train_sampler,
        num_workers=dataloader_worker,
        pin_memory=True,
    )

    model = BartForConditionalGeneration(config).cuda()
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    weight_decay = 0.02
    learning_rate = 5e-5
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            # Parameters that are not bias and LayerNorm.weight will receive a penalty
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=learning_rate)

    check_val_every_n_epoch = 1
    max_epochs = 5
    warmup_ratio = 0.3  # for scheduler
    # num_train_steps_per_epoch = int(len(train_dataloader) / (per_device_train_batch_size )) + int(len(val_dataloader) / (per_device_val_batch_size )) * check_val_every_n_epoch
    num_train_steps_per_epoch = int(
        len(processed_dataset["train"]) / (per_device_train_batch_size)
    )
    num_train_steps_per_epoch = (
        num_train_steps_per_epoch // world_size
    )  # for multiprocessing
    total_num_train_steps = num_train_steps_per_epoch * max_epochs

    num_warmup_steps = int(total_num_train_steps * warmup_ratio)
    gradient_accumulation_steps = 1
    # linear ->  get_linear_schedule_with_warmup,
    # cosine ->  get_cosine_schedule_with_warmup
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=total_num_train_steps,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
    )

    import wandb

    # wandb
    wandb.init(
        project="bart_pretraining-project",
        config={
            "dataset": f"{dataset_name}",
            "epochs": max_epochs,
            "learning-rate": learning_rate,
        },
    )
    ### Training Start ###
    train_dataloader_len = len(train_dataloader)
    global_count = 0
    model.train()
    for epoch in range(max_epochs):

        epoch_tqdm_dataloader = tqdm(
            train_dataloader,
            f"Training( {epoch} / {max_epochs} ) ",
            dynamic_ncols=True,
        )
        for i, batch in enumerate(epoch_tqdm_dataloader):
            global_count += 1
            optimizer.zero_grad()
            batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if i % 100 == 0:
                epoch_tqdm_dataloader.set_postfix(
                    {"loss": loss, "global_count": global_count}
                )
                wandb.log({"loss": loss})
            # model save
            if global_count % 10000 == 0:
                if rank == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        f"outputs/kobart_pre_epoch{epoch}_global_{global_count}_loss={loss:.5f}.pt",
                    )

    wandb.finish()
