from functools import partial

from datasets import load_dataset
from tasks.abstract_task import AbstractTask
from transformers import AutoModelForCausalLM


class CausalLMTask(AbstractTask):
    @staticmethod
    def get_model_class():
        return AutoModelForCausalLM.from_pretrained

    @staticmethod
    def get_inference_sample(tokenizer):
        return "I don't want a lot for Christmas. There is just one thing"

    @staticmethod
    def get_inference_output(tokenizer, output):
        return tokenizer.decode(output[0])

    @staticmethod
    def get_training_dataset():
        return load_dataset("squad").data["train"]["context"]

    @staticmethod
    def get_training_preprocessing(train_step, dataset):
        return dataset[:train_step]

    def get_training_inputs(self, sample, batch_size, max_length, tokenizer):
        inputs = self.tokenize(sample, batch_size, max_length, tokenizer)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"],
        }

    @staticmethod
    def name():
        return "causal_lm"

    @staticmethod
    def forward(model):
        return partial(model.generate, num_beams=3)
