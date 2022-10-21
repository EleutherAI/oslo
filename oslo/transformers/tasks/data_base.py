from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from datasets.arrow_dataset import Batch

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class BaseProcessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._chunk_size = max_length
        self._buffer = []

    def save_tokenizer(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)

    @abstractmethod
    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        pass


def pad_labels(
    labels,
    tokenizer,
    label_pad_token_id: int,
    pad_to_multiple_of: Optional[int] = None,
):
    labels = tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_attention_mask=False,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of,
    )["input_ids"]

    labels.masked_fill_(labels == tokenizer.pad_token_id, label_pad_token_id)
    return labels
