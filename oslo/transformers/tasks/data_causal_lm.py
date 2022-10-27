import logging
import warnings
from typing import Dict, List
from transformers import PreTrainedTokenizerBase
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForCausalLM(BaseProcessor):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self._chunk_size = max_length

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )

                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])

                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class DataCollatorForCausalLM(object):
    """
    Processing training examples to mini-batch for Gpt2 (clm).
    """

    def __init__(self, processor: ProcessorForCausalLM, label_pad_token_id: int = -100):
        if not isinstance(processor, ProcessorForCausalLM):
            warnings.warn(
                "DataCollatorForCausalLM is suitable for ProcessorForCausalLM."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in the processor._tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            examples, return_attention_mask=False, return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
