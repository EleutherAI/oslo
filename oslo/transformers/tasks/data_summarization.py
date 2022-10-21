import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor, pad_labels

try:
    from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForSummarization(BaseProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        super().__init__(tokenizer=tokenizer, max_length=max_length)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "labels" in column_names
        ), "The name of dataset column that you want to use as a summary must be 'labels'"

        assert (
            "text" in column_names
        ), "The name of dataset column that you want to use as a text must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {}

        dict_of_training_examples["input_ids"] = self._tokenizer(
            examples["text"],
            truncation=True,
            max_length=self._max_length,
            verbose=False,
        )["input_ids"]

        dict_of_training_examples["labels"] = self._tokenizer(
            examples["labels"],
            truncation=True,
            max_length=self._max_length,
            verbose=False,
        )["input_ids"]

        return dict_of_training_examples


class DataCollatorForSummarization(DataCollatorForSeq2Seq):
    """
    Processing training examples to mini-batch (summarization).
    """

    def __init__(
        self,
        processor: ProcessorForSummarization,
        model: Optional[Any] = None,
        label_pad_token_id: int = -100,
    ):
        if not isinstance(processor, ProcessorForSummarization):
            warnings.warn(
                "DataCollatorForSummarization is suitable for ProcessorForSummarization."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer
        self.model = model
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            return_attention_mask=True,
            return_tensors=None,
        )

        batch["labels"] = pad_labels(
            [feature["labels"] for feature in features],
            self.tokenizer,
            label_pad_token_id=self.label_pad_token_id,
        )

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        # prepare decoder_input_ids
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch["labels"]
            )
            batch["decoder_input_ids"] = decoder_input_ids
            batch["decoder_attention_mask"] = torch.where(
                decoder_input_ids == self.tokenizer.pad_token_id,
                0,
                torch.ones_like(decoder_input_ids),
            )

        return batch
