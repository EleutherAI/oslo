import time
import torch
from torch.utils.data import DataLoader
from datasets import Dataset as Dt
import os
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)
from tests_deprecated.transformers.tasks.test_data_base import TestDataBinarization
from transformers import BartTokenizerFast

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataBartPretraining(TestDataBinarization):
    def __init__(
        self,
        tokenizer,
        model_name="facebook/bart-base",
        label_pad_token_id=-100,
        max_seq_length=512,
    ):
        self.processor = ProcessorForBartPretraining(
            tokenizer, max_seq_length=max_seq_length
        )
        self.data_collator = DataCollatorForBartPretraining(
            self.processor,
            label_pad_token_id=label_pad_token_id,
            permute_sentence_ratio=0.5,
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer

    def __call__(
        self,
        dataset,
        mlm_probability=0.15,
        possion_lambda=3,
        batch_size=1024,
        batch_check_num_sample=2,
        batch_check_tokens=False,
    ):

        self.data_collator.mlm_probability = mlm_probability
        self.data_collator.possion_lambda = possion_lambda

        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Seq Length: {self.processor._max_seq_length}",
            f"Batch size: {batch_size}",
            f"MLM probability: {mlm_probability}",
            f"Possion Lambda: {possion_lambda}",
            sep="\n",
        )
        processed_dataset = self.processor(dataset)

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer = self.data_collator.tokenizer
            print("pad_token is set.")

        dataloader = DataLoader(
            processed_dataset["train"],
            batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        batch = next(iter(dataloader))
        self._batch_check(
            batch, num_samples=batch_check_num_sample, check_token=batch_check_tokens
        )

        self._length_check(
            dataloader,
            "input_ids",
            self.processor._max_seq_length,
            must_be_equal_to_max_length=False,
        )

        self._length_check(
            dataloader,
            "labels",
            self.processor._max_seq_length,
            must_be_equal_to_max_length=True,
        )

        self.mask_ratio_check(dataloader)

        print("---------- Test Pass ----------\n")

    def mask_ratio_check(self, dataloader):
        mask_token_id = self.tokenizer.mask_token_id
        pad_token_id = self.tokenizer.pad_token_id
        cnt, accum_mlm_prob = 0, 0
        for batch in dataloader:
            cnt += 1
            batch_size_label, seq_length_label = batch["labels"].shape
            batch_size_input, seq_length_input = batch["input_ids"].shape

            # Verify that the mask token ratio is aligned to a predetermined percentage
            num_pad_tokens = torch.sum(batch["input_ids"] == pad_token_id)
            num_mask_tokens = torch.sum(batch["input_ids"] == mask_token_id)
            num_labels = batch_size_label * (seq_length_label - 1)
            num_input_ids = batch_size_input * (seq_length_input - 2)
            mlm_probability = 1 - (num_input_ids - num_mask_tokens - num_pad_tokens) / (
                num_labels
            )
            assert torch.isclose(
                mlm_probability,
                torch.tensor(self.data_collator.mlm_probability),
                atol=0.005,
            ), f"Mask ratio({mlm_probability:.6f}) is different from the predefined one({self.data_collator.mlm_probability})"
            accum_mlm_prob += mlm_probability
        avg_mlm_prob = accum_mlm_prob / cnt
        print(f"MLM Probability: {avg_mlm_prob:.6f}")
        print("---- mask ratio test pass ----\n")


if "__main__" == __name__:

    # dataset = load_dataset("glue", "sst2")
    dataset = load_dataset(
        # data_args.dataset_name, ############ !!!!!!!!!!!!!!!!!!
        "cnn_dailymail",
        "3.0.0",
        use_auth_token=None,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    dataset = dataset.rename_column("article", "text")
    # reduce dataset for time
    dataset["train"] = Dt.from_dict(dataset["train"][:4523])
    dataset["validation"] = Dt.from_dict(dataset["train"][:555])
    dataset["test"] = Dt.from_dict(dataset["train"][:555])
    print(os.environ["MASTER_PORT"])
    model_nm = "facebook/bart-base"
    tokenizer = BartTokenizerFast.from_pretrained(model_nm)
    # parallel_context = ParallelContext.from_torch(sequence_parallel_size=1)
    bart_test = TestDataBartPretraining(
        tokenizer, model_name=model_nm, label_pad_token_id=-100, max_seq_length=256
    )
    bart_test(dataset, batch_size=4, batch_check_num_sample=2)
