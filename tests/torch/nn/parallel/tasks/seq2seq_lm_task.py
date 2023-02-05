from functools import partial

from datasets import load_dataset
from tasks.abstract_task import AbstractTask
from transformers import AutoModelForSeq2SeqLM


class Seq2SeqLMTask(AbstractTask):
    @staticmethod
    def get_model_class():
        return AutoModelForSeq2SeqLM.from_pretrained

    @staticmethod
    def get_inference_sample(tokenizer):
        return (
            "Life was like a box of chocolates. You never know what you’re gonna get."
        )

    @staticmethod
    def get_inference_output(tokenizer, output):
        return tokenizer.decode(output[0])

    @staticmethod
    def get_training_dataset():
        return load_dataset("wmt14", "de-en").data["train"][0]

    @staticmethod
    def get_training_preprocessing(train_step, dataset):
        return [(str(data[1]), str(data[0])) for data in dataset[:train_step]]

    def get_training_inputs(self, sample, batch_size, max_length, tokenizer):
        src = self.tokenize(sample[0], batch_size, max_length, tokenizer)
        tgt = self.tokenize(sample[1], batch_size, max_length, tokenizer)

        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": tgt["input_ids"],
        }

    @staticmethod
    def name():
        return "seq2seq_lm"

    @staticmethod
    def forward(model):
        return partial(model.generate, num_beams=3)
