import logging
import warnings
from typing import Any, Dict, List

from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor

try:
    from transformers import PreTrainedTokenizerBase
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForSequenceClassification(BaseProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        is_text_pair: bool = False,
    ) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)
        self.is_text_pair = is_text_pair

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "labels" in column_names
        ), "The name of dataset column that you want to use as a label must be 'labels'"

        if self.is_text_pair:
            assert (
                "text1" in column_names and "text2" in column_names
            ), "The name of dataset columns that you want to tokenize must be 'text1' and 'text2'"

            dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
                examples["text1"],
                examples["text2"],
                truncation=True,
                max_length=self._max_length,
                verbose=False,
            )
        else:
            assert (
                "text" in column_names
            ), "The name of dataset column that you want to tokenize must be 'text'"

            dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=self._max_length,
                verbose=False,
            )

        dict_of_training_examples["labels"] = examples["labels"]

        return dict_of_training_examples


class DataCollatorForSequenceClassification(object):
    """
    Processing training examples to mini-batch for Sequence Classification.
    """

    def __init__(self, processor: ProcessorForSequenceClassification):
        if not isinstance(processor, ProcessorForSequenceClassification):
            warnings.warn(
                "DataCollatorForSequenceClassification is suitable for ProcessorForSequenceClassification."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.tokenizer.pad(
            features,
            return_attention_mask=True,
            return_tensors="pt",
        )
