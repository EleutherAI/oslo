import logging

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
from oslo.transformers.trainer import Trainer
from oslo.transformers.training_args import TrainingArguments

logging.basicConfig(level=logging.INFO)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 데이터셋 생성
dataset = load_dataset("glue", "cola")
dataset = dataset.rename_column("sentence", "text")
dataset = dataset.rename_column("label", "labels")

processor = ProcessorForSequenceClassification(tokenizer, 512)
if processor._tokenizer.pad_token is None:
    processor._tokenizer.pad_token = processor._tokenizer.eos_token

processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()
train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

data_collator = DataCollatorForSequenceClassification(processor)

# Define trainer arguments
args = TrainingArguments(
    output_dir="output",
    eval_steps=500,
    optim="adam",
    lr_scheduler_type="linear",
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)

# Define trainer
trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# # Save
# trainer.save_model()

# Eval
metrics = trainer.evaluate(eval_dataset=valid_dataset)
