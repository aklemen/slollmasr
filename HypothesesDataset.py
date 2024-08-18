import torch
import pandas as pd
import numpy as np
import logging
from transformers import AutoTokenizer


class HypothesesDataset(torch.utils.data.Dataset):
    def __init__(self, hypotheses: pd.DataFrame, ground_truths: list[str], tokenizer: AutoTokenizer, beam_size: int, max_seq_length: int):
        self.hypotheses = hypotheses
        self.ground_truths = ground_truths
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length

        self.pad_id = self._get_pad_id(tokenizer)

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

    def get_beam_size(self):
        return self.beam_size

    def get_ground_truth_for_hypothesis_at(self, idx: int) -> str:
        return self.ground_truths[idx // self.beam_size]

    def _convert_text_to_ids(self, hypothesis_text):
        tokens = self.tokenizer.tokenize(hypothesis_text)
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if hasattr(self.tokenizer, "bos_id") and self.tokenizer.bos_id is not None:
            hypothesis_ids = [self.tokenizer.bos_id] + hypothesis_ids
        if hasattr(self.tokenizer, "eos_id") and self.tokenizer.eos_id is not None:
            hypothesis_ids = hypothesis_ids + [self.tokenizer.eos_id]
        return hypothesis_ids

    def _get_pad_id(self, tokenizer):
        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id is not None:
            return tokenizer.pad_id
        elif hasattr(tokenizer, "eos_id") and tokenizer.eos_id is not None:
            return tokenizer.eos_id
        else:
            logging.info(f"Using 0 as pad_id as the tokenizer has no pad_id or eos_id.")
            return 0

    def _get_input_ids(self, hypothesis_ids):
        input_ids = [self.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        input_ids = np.array(input_ids)
        return input_ids

    def _get_input_mask(self, hypothesis_ids):
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
