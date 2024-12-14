import editdistance
import pandas as pd
import numpy as np

from Tokenizer import Tokenizer
from torch_datasets.HypothesesDataset import HypothesesDataset


class HypothesesWithIdsDataset(HypothesesDataset):
    def __init__(
            self,
            hypotheses: pd.DataFrame,
            ground_truths: list[str],
            tokenizer: Tokenizer,
            max_seq_length: int = 1024,
    ):
        super().__init__(hypotheses, ground_truths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        hypothesis = self._get_hypothesis_text(idx)
        asr_score = self._get_hypothesis_score(idx)
        hypothesis_ids = self.tokenizer.text_to_ids(hypothesis)
        input_ids = self._get_input_ids(hypothesis_ids)
        input_mask = self._get_input_mask(hypothesis_ids)
        char_length = len(hypothesis)
        ground_truth = self.ground_truths[idx // self.beam_size]
        distance_to_ground_truth = editdistance.eval(hypothesis.split(), ground_truth.split())
        return hypothesis, asr_score, input_ids, input_mask, char_length, distance_to_ground_truth

    def _get_input_ids(self, hypothesis_ids):
        input_ids = [self.tokenizer.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        input_ids = np.array(input_ids)
        return input_ids

    def _get_input_mask(self, hypothesis_ids):
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
