from functools import partial

import torch
from datasets import load_dataset
from tasks.abstract_task import AbstractTask
from transformers import AutoModelForSequenceClassification


class SequenceClassificationTask(AbstractTask):
    @staticmethod
    def get_model_class():
        return partial(AutoModelForSequenceClassification.from_pretrained, num_labels=3)

    @staticmethod
    def get_inference_sample(tokenizer):
        return "I will decide how I feel, I will be happy today."

    @staticmethod
    def get_inference_output(tokenizer, output):
        return output.logits.argmax(-1).item()

    @staticmethod
    def get_training_dataset():
        return load_dataset("multi_nli").data["train"]

    @staticmethod
    def get_training_preprocessing(train_step, dataset):
        return [
            (f"{str(p)}\n{str(h)}", l.as_py())
            for p, h, l in list(zip(dataset[2], dataset[5], dataset[9]))[:train_step]
        ]

    def get_training_inputs(self, sample, batch_size, max_length, tokenizer):
        inputs = self.tokenize(sample[0], batch_size, max_length, tokenizer)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(sample[1])
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to("cuda"),
        }

    @staticmethod
    def name():
        return "sequence_classification"

    @staticmethod
    def forward(model):
        return model.forward
