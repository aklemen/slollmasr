import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from Tokenizer import Tokenizer


class HypothesesDataset(Dataset):
    def __init__(self, hypotheses: pd.DataFrame, tokenizer: Tokenizer, beam_size: int, max_seq_length: int = 1024):
        self.hypotheses = hypotheses
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, idx):
        hypothesis = str(self.hypotheses["text"][idx])
        asr_score = self.hypotheses["score"][idx]
        hypothesis_ids = self.tokenizer.text_to_ids(hypothesis)
        input_ids = self._get_input_ids(hypothesis_ids)
        input_mask = self._get_input_mask(hypothesis_ids)
        return hypothesis, asr_score, input_ids, input_mask

    def get_hypotheses_texts(self) -> list[str]:
        return self.hypotheses["text"].tolist()

    def get_hypotheses_scores(self) -> list[float]:
        return self.hypotheses["score"].tolist()

    def get_beam_size(self):
        return self.beam_size

    def get_number_of_samples(self):
        return len(self.hypotheses) // self.beam_size

    def _get_input_ids(self, hypothesis_ids):
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        input_ids = np.array(input_ids)
        return input_ids

    def _get_input_mask(self, hypothesis_ids):
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
