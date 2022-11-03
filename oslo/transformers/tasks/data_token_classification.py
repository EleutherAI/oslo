import logging
import warnings
from typing import Dict, List, Union, Any

import torch
from datasets import Dataset, DatasetDict
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor

try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerBase,
        RobertaTokenizer,
    )
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForTokenClassification(BaseProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        dataset: Union[Dataset, DatasetDict] = None,
    ) -> None:
        super().__init__(tokenizer, max_length)
        if dataset is None:
            raise ValueError(
                "dataset argument must be set. (dataset: Union[Dataset, DatasetDict])"
            )

        if self._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.label_names = self.get_label_names(dataset)
        self.label_map = self.get_label_map(self.label_names)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "tokens" in column_names
        ), "The name of dataset column that you want to tokenize must be 'tokens' (not 'text')"

        dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=self._max_length,
            is_split_into_words=True,
            verbose=False,
        )
        all_labels = examples["labels"]
        new_labels = []
        self.make_B_to_I_label(self.label_names)

        for i, labels in enumerate(all_labels):
            word_ids = dict_of_training_examples.word_ids(batch_index=i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        dict_of_training_examples["labels"] = new_labels
        return dict_of_training_examples

    def get_label_names(self, dataset: Union[Dataset, DatasetDict]) -> List[str]:
        if isinstance(dataset, Dataset):
            assert (
                "labels" in dataset.features
            ), "The name of dataset column that you want to use as a label must be 'labels'"

            features = dataset.features["labels"]
            label_names = features.feature.names
        else:
            assert (
                "train" in dataset.keys()
            ), "The key name of train dataset must be 'train'"
            assert (
                "labels" in dataset["train"].features
            ), "The name of dataset column that you want to use as a label must be 'labels'"

            features = dataset["train"].features["labels"]
            label_names = features.feature.names

        self.label_names = label_names

        return label_names

    def get_label_map(
        self, label_names: Union[List[str], Dataset, DatasetDict]
    ) -> Dict[str, Dict[str, str]]:
        if isinstance(label_names, Dataset) or isinstance(label_names, DatasetDict):
            label_names = self.get_label_names(label_names)

        id2label = {str(i): label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}

        self.id2label = id2label
        self.label2id = label2id

        return {"label2id": label2id, "id2label": id2label}

    def make_B_to_I_label(self, label_names: List[str]) -> List[int]:
        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_names):
            if label.startswith("B-") and label.replace("B-", "I-") in label_names:
                b_to_i_label.append(label_names.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        self.b_to_i_label = b_to_i_label

        return b_to_i_label

    def align_labels_with_tokens(self, labels, word_ids) -> List[int]:
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to pad token id
            if word_idx is None:
                label_ids.append(self._tokenizer.pad_token_id)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(self.b_to_i_label[labels[word_idx]])
            previous_word_idx = word_idx

        return label_ids


class DataCollatorForTokenClassification(object):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """

    def __init__(
        self,
        processor: ProcessorForTokenClassification,
        padding_side: str = "right",
        label_pad_token_id: int = -100,
    ):
        if not isinstance(processor, ProcessorForTokenClassification):
            warnings.warn(
                "DataCollatorForTokenClassification is suitable for ProcessorForTokenClassification."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer
        self.padding_side = padding_side
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]):
        label_name = "labels"
        labels = [feature[label_name] for feature in features]
        return self.pad(features, labels, label_name)

    def pad(self, features, labels, label_name):
        batch = self.tokenizer.pad(
            features,
            return_attention_mask=True,
            return_tensors=None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch[label_name].masked_fill_(
            batch[label_name] == self.tokenizer.pad_token_id, self.label_pad_token_id
        )
        return batch
