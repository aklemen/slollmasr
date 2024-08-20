import torch
import pandas as pd
import numpy as np
import logging
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union


class HypothesesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hypotheses: pd.DataFrame,
            ground_truths: list[str],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            beam_size: int,
            max_seq_length: int
    ):
        self.hypotheses = hypotheses
        self.ground_truths = ground_truths
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length

        self.pad_id = self._get_pad_id(tokenizer)
        self.bos_id = self._get_bos_id(tokenizer)
        self.eos_id = self._get_eos_id(tokenizer)

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, idx):
        ground_truth = self.get_ground_truth_for_hypothesis_at(idx)
        hypothesis = str(self.hypotheses["text"][idx])
        asr_score = self.hypotheses["score"][idx]
        hypothesis_ids = self._convert_text_to_ids(hypothesis)
        input_ids = self._get_input_ids(hypothesis_ids)
        input_mask = self._get_input_mask(hypothesis_ids)
        return ground_truth, hypothesis, asr_score, input_ids, input_mask

    def get_hypotheses_texts(self) -> list[str]:
        return self.hypotheses["text"].tolist()

    def get_hypotheses_scores(self) -> list[float]:
        return self.hypotheses["score"].tolist()

    def get_beam_size(self):
        return self.beam_size

    def get_number_of_samples(self):
        return len(self.ground_truths)

    def get_ground_truth_for_hypothesis_at(self, idx: int) -> str:
        return self.ground_truths[idx // self.beam_size]

    def _convert_text_to_ids(self, hypothesis_text):
        tokens = self.tokenizer.tokenize(hypothesis_text)
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.bos_id is not None:
            hypothesis_ids = [self.bos_id] + hypothesis_ids
        if self.eos_id is not None:
            hypothesis_ids = hypothesis_ids + [self.eos_id]
        return hypothesis_ids

    def _get_pad_id(self, tokenizer):
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None:
            return tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        elif hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            logging.info(f"Using eos_id as pad_id as the tokenizer has no pad_token.")
            return tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
        else:
            logging.info(f"Using 0 as pad_id as the tokenizer has no pad_token or eos_token.")
            return 0

    def _get_bos_id(self, tokenizer):
        if getattr(tokenizer, 'bos_token') is None:
            return None
        return tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]

    def _get_eos_id(self, tokenizer):
        if getattr(tokenizer, 'eos_token') is None:
            return None
        return tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

    def _get_input_ids(self, hypothesis_ids):
        input_ids = [self.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        input_ids = np.array(input_ids)
        return input_ids

    def _get_input_mask(self, hypothesis_ids):
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
