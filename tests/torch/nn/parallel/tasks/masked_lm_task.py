from tasks.abstract_task import AbstractTask
from transformers import AutoModelForMaskedLM


class MaskedLMTask(AbstractTask):
    @staticmethod
    def get_model_class():
        return AutoModelForMaskedLM.from_pretrained

    @staticmethod
    def get_inference_sample(tokenizer):
        return f"Manners maketh man. Do you {tokenizer.mask_token} what that means?"

    @staticmethod
    def get_inference_output(tokenizer, output):
        return tokenizer.decode(output.logits.argmax(-1)[0])

    @staticmethod
    def get_training_dataset():
        raise NotImplementedError

    @staticmethod
    def get_training_preprocessing(train_step, dataset):
        raise NotImplementedError

    def get_training_inputs(self, sample, batch_size, max_length, tokenizer):
        raise NotImplementedError

    @staticmethod
    def name():
        return "masked_lm"

    @staticmethod
    def forward(model):
        return model.forward
