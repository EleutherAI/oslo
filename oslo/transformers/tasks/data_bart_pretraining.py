import logging
import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor

try:
    from transformers import BartTokenizer, BartTokenizerFast, PreTrainedTokenizerBase
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logging.captureWarnings(True)


class ProcessorForBartPretraining(BaseProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ) -> None:
        super().__init__(tokenizer, max_length)

        if not isinstance(self._tokenizer, (BartTokenizer, BartTokenizerFast)):
            warnings.warn(
                "PorcessorForBartPretraining is only suitable for BartTokenizer-like tokenizers."
            )

        self._chunk_size = max_length - 1

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.sep_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class DataCollatorForBartPretraining(object):
    """
    Processing training examples to mini-batch for Bart (text_infilling, sentence_permutation).
    """

    def __init__(
        self,
        processor: ProcessorForBartPretraining,
        mlm_probability: float = 0.15,
        possion_lambda: float = 3.0,
        permute_sentence: bool = True,
        label_pad_token_id: int = -100,
        decoder_start_token_id: Optional[int] = None,
    ):
        if mlm_probability >= 1.0:
            warnings.warn("MLM Probability is greater than 1.0")

        assert isinstance(
            processor, ProcessorForBartPretraining
        ), "DataCollatorForBartPretraining is only suitable for ProcessorForBartPretraining."

        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.possion_lambda = possion_lambda
        self.permute_sentence = permute_sentence
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = processor._tokenizer.mask_token_id
        self.label_pad_token_id = label_pad_token_id
        self.decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id
            else self.tokenizer.eos_token_id
        )

        self.get_start_indices = {
            max_idx: np.random.choice(max_idx, size=(max_idx,), replace=False)
            for max_idx in range(processor._chunk_size, 0, -1)
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_noise_text_from_examples(examples)
        batch = self.tokenizer.pad(
            examples,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return self._prepare_decoder_inputs_from_labels(batch)

    def _prepare_noise_text_from_examples(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            labels = chunk_ids[:]

            chunk_ids = self._text_infilling(chunk_ids)
            if self.permute_sentence:
                chunk_ids = self._sentence_permutation(chunk_ids)

            chunk_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)
            labels = self.tokenizer.build_inputs_with_special_tokens(labels)[1:]

            output_examples.append(
                {
                    "input_ids": chunk_ids,
                    "labels": labels,
                }
            )

        return output_examples

    def _text_infilling(self, input_ids: List[int]) -> List[int]:
        length = len(input_ids)
        num_noise_tokens = int(np.round(length * self.mlm_probability))

        # pick the lengths of the noise spans
        def _possion_segmentation(num_noise_tokens):
            segment_lengths = []
            while sum(segment_lengths) < num_noise_tokens:
                span_length = np.random.poisson(lam=self.possion_lambda)
                segment_lengths.append(span_length)

            difference = sum(segment_lengths) - num_noise_tokens
            segment_lengths[-1] = segment_lengths[-1] - difference
            segment_lengths.sort(reverse=True)
            return segment_lengths

        temp_ids = input_ids
        while len(temp_ids) >= length:
            temp_ids = input_ids[:]
            noise_span_lengths = _possion_segmentation(num_noise_tokens)

            for noise_span_length in noise_span_lengths:
                max_idx = len(temp_ids) - noise_span_length + 1
                # get start index of mask span
                start_indices = self.get_start_indices[max_idx]
                for start_idx in start_indices:
                    if (
                        self.mask_token_id
                        in temp_ids[start_idx : start_idx + noise_span_length]
                    ):
                        continue
                    else:
                        temp_ids = (
                            temp_ids[:start_idx]
                            + [self.mask_token_id]
                            + temp_ids[start_idx + noise_span_length :]
                        )
                        # rotate start indices
                        self.get_start_indices[max_idx] = np.roll(start_indices, 1)
                        break

        input_ids = temp_ids

        return input_ids

    def _sentence_permutation(self, input_ids: List[int]) -> List[int]:
        ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        split_sentences = []
        split_points = [
            idx for idx, token in enumerate(ref_tokens) if token in (".", "Ġ.")
        ]

        if split_points:
            prev_point = 0
            for split_point in split_points:
                split_point += 1
                split_sentences.append(input_ids[prev_point:split_point])
                prev_point = split_point
            split_sentences.append(input_ids[prev_point:])

            random.shuffle(split_sentences)

            input_ids = []
            for split_sentence in split_sentences:
                input_ids.extend(split_sentence)

        return input_ids

    def _prepare_decoder_inputs_from_labels(self, batch):
        # Shift input ids one token to the right
        shifted_input_ids = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_input_ids[:, 1:] = batch["labels"][:, :-1].clone()
        shifted_input_ids[:, 0] = self.decoder_start_token_id

        shifted_input_ids.masked_fill_(
            shifted_input_ids == self.label_pad_token_id,
            self.pad_token_id,
        )

        batch["decoder_input_ids"] = shifted_input_ids
        batch["decoder_attention_mask"] = torch.where(
            shifted_input_ids == self.pad_token_id,
            0,
            torch.ones_like(shifted_input_ids),
        )
        return batch
