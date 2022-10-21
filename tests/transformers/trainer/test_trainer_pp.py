###################################################
# WARNING: This script not work
###################################################

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from oslo.transformers.training_args import TrainingArguments
from oslo.transformers.trainer import Trainer
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
import logging

logging.basicConfig(level=logging.INFO)

os.environ["WANDB_DISABLED"] = "true"

oslo_init_dict_form = {
    "data_parallelism": {
        "enable": False,
        "parallel_size": 2,
        "zero_stage": 2,
    },
    "tensor_parallelism": {
        "enable": False,
        "parallel_size": 4,
        "parallel_mode": "3d",
    },
    "pipeline_parallelism": {"enable": True, "parallel_size": 2},
}
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

processor = ProcessorForSequenceClassification(tokenizer, 512)
if processor._tokenizer.pad_token is None:
    processor._tokenizer.pad_token = processor._tokenizer.eos_token

# 데이터셋 생성
dataset = load_dataset("glue", "cola")
dataset = dataset.rename_column("sentence", "text")
dataset = dataset.rename_column("label", "labels")

processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()
train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

data_collator = DataCollatorForSequenceClassification(processor)


args = TrainingArguments(
    output_dir="output",
    eval_steps=500,
    optim="adam",
    lr_scheduler_type="linear",
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    oslo_config_path_or_dict=oslo_init_dict_form,
    dataloader_drop_last=True,
)

trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()
