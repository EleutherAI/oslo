import torch
from torch.utils.data import DataLoader

from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_permutation_lm import (
    ProcessorForPermutationLM,
    DataCollatorForPermutationLM,
)
from tests_deprecated.transformers.tasks.test_data_base import TestDataBinarization

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataPermutationLM(TestDataBinarization):
    def __init__(
        self, 
        model_name,
        parallel_context=None, 
        label_pad_token_id=-100
    ):
        self.processor = ProcessorForPermutationLM(model_name)
        self.data_collator = DataCollatorForPermutationLM(
            self.processor, label_pad_token_id=label_pad_token_id
        )
        self.sp_data_collator = DataCollatorForPermutationLM(
            self.processor,
            parallel_context=parallel_context,
            label_pad_token_id=label_pad_token_id,
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.parallel_context = parallel_context

    def __call__(
        self,
        max_length,
        dataset,
        plm_probability=1/6,
        max_span_length=5,
        batch_size=1024,
        batch_check_num_sample=2,
        batch_check_tokens=False,
    ):
        self.processor._chunk_size = max_length - 1 #max_length
        self.processor._max_length = max_length
        self.data_collator.plm_probability = plm_probability

        if self.sp_data_collator:
            self.sp_data_collator.plm_probability = plm_probability
            self.sp_data_collator.max_span_length = max_span_length
        additional_special_ids = self.tokenizer.additional_special_tokens_ids
        min_additional_special_id = min(additional_special_ids)

        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {max_length}",
            f"Batch size: {batch_size}",
            f"PLM probability: {plm_probability}",
            f"Max span length: {max_span_length}\n",
            sep="\n",
        )
        processed_dataset = dataset.map(
            self.processor, batched=True, remove_columns=dataset["train"].column_names
        )

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
            batch,
            num_samples=batch_check_num_sample,
            check_token=batch_check_tokens,
            additional_special_ids=additional_special_ids,
        )

        self._length_check(
            dataloader,
            "input_ids",
            max_length,
            must_be_equal_to_max_length=False,
        )

        self._length_check(
            dataloader,
            "labels",
            self.processor.target_chunk_size,
            must_be_equal_to_max_length=False,
        )

        self.mask_ratio_check(dataloader, min_additional_special_id)

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)

        print("---------- Test Pass ----------\n")

    def _batch_check(
        self, batch, num_samples, check_token, additional_special_ids
    ) -> None:
        print("--------- batch check ---------\n")
        print(f"batch keys: {', '.join([key for key in batch.keys()])}\n")
        for key, value in batch.items():
            print(f"{key} size: {value.size()}")

        for idx in range(num_samples):
            for key, value in batch.items():
                print(f"{key}: \n{value[idx]}\n")

            for key, value in batch.items():
                if key == "input_ids":
                    print(f"input_ids decode: \n{self.tokenizer.decode(value[idx])}\n")
                    input_ids = value[idx]
                elif key == "labels" and value.dim() != 1:
                    if torch.any(value[idx] < 0):
                        continue
                    print(f"labels decode: \n{self.tokenizer.decode(value[idx])}\n")
                    labels = value[idx]

            text = []
            for input_id in input_ids:
                if input_id not in additional_special_ids:
                    text.append(input_id)
                else:
                    for label_idx, label_id in enumerate(labels):
                        if label_idx == 0:
                            continue
                        if label_id not in additional_special_ids:
                            text.append(label_id)
                        else:
                            labels = labels[label_idx:]
                            break

            print(f"text: \n{self.tokenizer.decode(text)}\nlength: {len(text)}\n")

            if check_token:
                print(
                    f"tokens: \n{self.tokenizer.convert_ids_to_tokens(batch['input_ids'][idx])}\n"
                )

    def _test_sp_collator(
        self,
        processed_dataset,
        batch_size,
    ):
        local_world_size = self.sp_data_collator.local_world_size

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print("pad_token is set.")

        if self.sp_data_collator.tokenizer.pad_token is None:
            self.sp_data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print("pad_token is set. (SP)")

        dataloader = DataLoader(
            processed_dataset["train"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        sp_dataloader = DataLoader(
            processed_dataset["train"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.sp_data_collator,
        )

        sp_batch = next(iter(sp_dataloader))
        print("(SP batch check)")
        print("--------- batch check ---------\n")
        print(f"batch keys: {', '.join([key for key in sp_batch.keys()])}\n")
        for key, value in sp_batch.items():
            print(f"{key} size: {value.size()}")

        for batch, sp_batch in zip(dataloader, sp_dataloader):
            seq_length = batch["input_ids"].size(1)
            sq_seq_length = sp_batch["input_ids"].size(1)

            if seq_length % sq_seq_length:
                sp_desired_length = (seq_length // local_world_size) + 1
                assert (
                    sp_desired_length == sq_seq_length
                ), f"Required length for SP({sp_desired_length} doesn't equal to SP sequence length({sq_seq_length}))"

        print("---- SP collator test pass ----\n")

    def mask_ratio_check(self, dataloader, min_additional_special_id):
        for batch in dataloader:
            batch_size, input_seq_length = batch["input_ids"].size()
            label_seq_length = batch["labels"].size(1)

            num_mask_span = torch.sum(batch["labels"] >= min_additional_special_id)
            num_input_ids = batch_size * (input_seq_length - 1) - num_mask_span
            num_labels = batch_size * (label_seq_length - 1) - num_mask_span
            num_total = num_input_ids + num_labels
            mlm_probability = num_labels / num_total
            assert torch.isclose(
                mlm_probability,
                torch.tensor(self.data_collator.plm_probability),
                atol=0.005,
            ), f"Mask ratio({mlm_probability:.6f}) is different from the predefined one({self.data_collator.plm_probability})"

        print(f"MLM Probability: {mlm_probability:.6f}")
        print("---- mask ratio test pass ----\n")


if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    plm_test = TestDataPermutationLM("xlnet-base-cased")
    # plm_test(512, dataset)
    # plm_test(511, dataset)
    # plm_test(253, dataset, 0.15, 4)
    # plm_test(128, dataset, batch_size=4)
    # plm_test(256, dataset, 0.2, batch_size=4)
    # plm_test(128, dataset, 0.3, batch_size=4)

    parallel_context = ParallelContext.from_torch(sequence_parallel_size=4)
    plm_sp_test = TestDataPermutationLM("xlnet-base-cased", parallel_context, 0)
    plm_sp_test(253, dataset)