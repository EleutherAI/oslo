import logging
import warnings
from typing import Any, Dict, List, Optional
import math
import numpy as np
import torch
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_base import BaseProcessor
import os
import torch.distributed as dist


try:
    from transformers import (
        BartTokenizer,
        BartTokenizerFast,
        PreTrainedTokenizerBase,
        BatchEncoding,
    )
    from transformers.models.bart.modeling_flax_bart import shift_tokens_right
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")
import re
from itertools import chain

logging.captureWarnings(True)


class ProcessorForBartPretraining(BaseProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 1024,
    ) -> None:
        super().__init__(tokenizer, max_seq_length)

        if not isinstance(self._tokenizer, (BartTokenizer, BartTokenizerFast)):
            warnings.warn(
                "PorcessorForBartPretraining is only suitable for BartTokenizer-like tokenizers."
            )

    def __call__(self, dataset) -> Dict[str, List[int]]:
        column_names = list(dataset["train"][0].keys())
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        def split_sentence_add_token(example):
            # l reprise the role in the last two films..  Watch I-Reporter give her review of Potter's latest » . There is life beyond
            pattern = r"(?<=[.!?])\s"
            splited_sents = [
                self._tokenizer.bos_token
                + f"{self._tokenizer.pad_token}".join(
                    [
                        sp_text.strip()
                        for sp_text in re.split(pattern, text)
                        if len(sp_text)
                        > 1  # Conditional statement for the last sentence
                    ]
                )
                + self._tokenizer.eos_token
                for text in example["text"]
            ]

            tokenized_sents = self._tokenizer(
                splited_sents,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=False,
                return_special_tokens_mask=False,
                verbose=False,
            )["input_ids"]

            return {"input_ids": tokenized_sents}

        # using maximum cpu power
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        os_count = os.cpu_count()
        split_tokenized_dataset = dataset.map(
            split_sentence_add_token,
            batched=True,
            num_proc=os_count // world_size,  #  number of multiprocessor
            remove_columns=column_names,
        )

        def group_text(examples):
            self._buffer.extend(
                list(chain.from_iterable(examples["input_ids"]))
            )  # combine input_ids dataset
            total_length = len(self._buffer)
            quotient, remain = divmod(total_length, self._max_seq_length)
            output = [
                self._buffer[i : i + self._max_seq_length]
                for i in range(0, quotient * self._max_seq_length, self._max_seq_length)
            ]
            self._buffer = self._buffer[quotient * self._max_seq_length :]
            return {"input_ids": output}

        group_dataset = split_tokenized_dataset.map(
            group_text, batched=True, num_proc=os_count // world_size
        )

        return group_dataset


class DataCollatorForBartPretraining(object):
    """
    Processing training examples to mini-batch for Bart (text_infilling, sentence_permutation).
    processor:
        pass
    mask_ratio (:obj:`float`):
        The probability with which to (randomly) mask tokens in the input
    poisson_lambda (:obj:`float`):
        Mean parameter of Poisson distribution used to generate span-lengths to be masked
    permute_sentence_ratio (:obj:`float`):
        Ratio of sentences to be permuted in each document
    decoder_start_token_id: (:obj:`int):
        The decoder start token id of the model
    label_pad_token_id:
        pass

    """

    def __init__(
        self,
        processor: ProcessorForBartPretraining,
        mask_ratio: float = 0.3,
        poisson_lambda: float = 3.0,
        permute_sentence_ratio: bool = 0.8,
        decoder_start_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):

        assert isinstance(
            processor, ProcessorForBartPretraining
        ), "DataCollatorForBartPretraining is only suitable for ProcessorForBartPretraining."

        self.tokenizer = processor._tokenizer
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.permute_sentence_ratio = permute_sentence_ratio
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
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )
        batch["labels"] = batch["input_ids"].copy()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        do_permute = False
        if self.permute_sentence_ratio:
            batch["input_ids"] = self.permute_sentences(batch["input_ids"])
            do_permute = True

        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.text_infilling(
                batch["input_ids"], batch["labels"], do_permute
            )

        # ignore pad tokens
        batch["attention_mask"] = (
            batch["input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        batch["decoder_attention_mask"] = (
            batch["decoder_input_ids"] != self.tokenizer.pad_token_id
        ).astype(int)
        return batch

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.copy()

        # find end locations of sentences
        # When creating an example, when a sentence comes in, sentence separation is performed, and the PAD token is inserted between the sentences.
        end_sentence_mask = input_ids == self.tokenizer.pad_token_id
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(
            sentence_ends[:, 0], return_counts=True
        )

        # E.X. num_sentences_map[3] = 10 indicates that there are 10 separate sentences in batch_index 3.
        num_sentences_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, num_sentences)
        }
        # Num_to_permute mixes only the sentence by the permit_ratio of the total when there are actually 10 sentences to be separated.
        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(
            int
        )
        num_to_permute_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, num_to_permute)
        }

        sentence_ends = np.split(
            sentence_ends[:, 1],
            np.unique(sentence_ends[:, 0], return_index=True)[1][1:],
        )
        sentence_ends_map = {
            sent_idx: count
            for sent_idx, count in zip(example_has_multiple_sentences, sentence_ends)
        }

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            # Create a random number of sentences to be separated (num_sentences_map) and extract by the permit_ratio (num_to_permute_map).
            # Substitutions is a random selection of sentences to be selected, and ordering[substitutions] is a random mixing of selected sentences.
            substitutions = np.random.permutation(num_sentences_map[i])[
                : num_to_permute_map[i]
            ]
            ordering = np.arange(0, num_sentences_map[i])
            # In orderring, mix the sentences by replacing the index corresponding to substitions with a random mixture of num_to_permute_map.
            ordering[substitutions] = substitutions[
                np.random.permutation(num_to_permute_map[i])
            ]

            # write shuffled sentences into results
            # Need to solve the problem of having to repeat the whole sentence, later.
            index = 0
            for j in ordering:
                sentence = input_ids[
                    i,
                    (sentence_ends_map[i][j - 1] if j > 0 else 0) : sentence_ends_map[
                        i
                    ][j],
                ]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def text_infilling(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """

        # get spectial_token in special_tokens of tokenizer
        f_get_special_tokens_mask = lambda x: self.tokenizer.get_special_tokens_mask(
            x, already_has_special_tokens=True
        )
        special_tokens_mask_labels = np.apply_along_axis(
            f_get_special_tokens_mask, 0, labels
        ).astype(bool)
        special_tokens_mask_inputs = np.apply_along_axis(
            f_get_special_tokens_mask, 0, input_ids
        ).astype(bool)

        # determine how many tokens we need to mask in total
        # masking something that is not a PAD token and not a Special token.
        is_token_mask = (
            ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask_inputs
        )
        num_tokens_to_mask = int(
            math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio)
        )
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(
            lam=self.poisson_lambda, size=(num_tokens_to_mask,)
        )
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [
                    span_lengths,
                    np.random.poisson(
                        lam=self.poisson_lambda, size=(num_tokens_to_mask,)
                    ),
                ]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        # Found index of span_length to select token as much as num_tokens_to_mask
        cutoff_idx = (
            np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        )
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask)
        span_starts = np.random.permutation(token_indices.shape[0])[
            : span_lengths.shape[0]
        ]
        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = np.full_like(input_ids, fill_value=False)

        # masking process
        # Put a True value in the index corresponding to the masked_indices,
        # do -1 in span_length since one mask token is in it, and calculate remaining whether there is anything left to mask.
        # The index of masked_indices is +1 to point to the next token.
        # Repeat the above process to insert the mask token until the span_length is all 0 or less and the masked_indices are less than max_index.

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1
        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            # Need to modify only the mainnig part repeatedly without repeating all masked_indices, B.S
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        # If the mask corresponds to special_tokens_mask_inputs, enter False because you should not mask
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_token_id
        if not do_permute:  # why this code needs?
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)
        for i, example in enumerate(input_ids):
            # remove all span_masked_tokens except for the first token
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels
