from abc import ABC


class AbstractTask(ABC):
    @staticmethod
    def get_model_class():
        raise NotImplementedError

    @staticmethod
    def get_inference_sample(tokenizer):
        raise NotImplementedError

    @staticmethod
    def get_inference_output(tokenizer, output):
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def forward(model):
        raise NotImplementedError

    @staticmethod
    def tokenize(sample, batch_size, max_length, tokenizer):
        return tokenizer(
            [str(sample)] * batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to("cuda")
