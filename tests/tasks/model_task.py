import torch
import os

from functools import partial
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ModelTask:
    def __init__(self):
        """
        Define model task
        """
        self.tasks = {
            "sequence-classification": {
                "class": partial(
                    AutoModelForSequenceClassification.from_pretrained, num_labels=3
                ),
                "load_dataset": load_dataset(
                    "pietrolesci/gpt3_nli", split="train", cache_dir="tests/cache"
                ),
                "preprocessing_map_func": self.mli_task_map_func,
            },
            "causal-lm": {
                "class": AutoModelForCausalLM.from_pretrained,
                "load_dataset": load_dataset(
                    "squad",
                    split="train",
                    cache_dir="tests/cache",
                ),
                "preprocessing_map_func": self.causal_lm_task_map_func,
            },
            "seq2seq": {
                "class": AutoModelForSeq2SeqLM.from_pretrained,
                "load_dataset": load_dataset(
                    "squad",
                    split="train",
                    cache_dir="tests/cache",
                ),
                "preprocessing_map_func": self.seq2seq_task_map_func,
            },
        }

    def get_model_task(self, task):

        assert task in self.tasks, (
            f"{task} is not supported task. "
            f"Please choose one of {list(self.tasks.keys())}. "
            "If there are no major problems, it will work for other tasks as well, "
            "but I haven't tested it, so if you encounter any problems, "
            "please report them through the github issue."
        )

        return self.tasks[task]

    def mli_task_map_func(self, dataset, tokenizer, args):
        def preprocess(row_datas):
            input_texts = []
            labels = []

            for text_a, text_b, label in zip(
                row_datas["text_a"], row_datas["text_b"], row_datas["label"]
            ):
                input_texts.append(f"{str(text_a)}\n{str(text_b)}")
                labels.append(label)

            input_text = tokenizer(
                input_texts,
                max_length=args.sequence_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            ret_labels = torch.tensor(labels, dtype=torch.long)

            return {**input_text, "labels": ret_labels}

        dataset = dataset.select(range(args.train_step))
        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=["text_a", "text_b", "label"],
        ).with_format("torch")

    def causal_lm_task_map_func(self, dataset, tokenizer, args):
        def preprocess(row_datas):
            input_text = tokenizer(
                row_datas["context"],
                max_length=args.sequence_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            return {**input_text, "labels": input_text["input_ids"]}

        dataset = dataset.select(range(args.train_step))

        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=["id", "title", "context", "question", "answers"],
        ).with_format("torch")

    def seq2seq_task_map_func(self, dataset, tokenizer, args):
        def preprocess(row_datas):
            input_text = tokenizer(
                row_datas["context"],
                max_length=args.sequence_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            label_text = tokenizer(
                row_datas["question"],
                max_length=args.sequence_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            return {**input_text, "labels": label_text["input_ids"]}

        dataset = dataset.select(range(args.train_step))

        return dataset.map(
            preprocess,
            batched=True,
            remove_columns=["id", "title", "context", "question", "answers"],
        ).with_format("torch")
