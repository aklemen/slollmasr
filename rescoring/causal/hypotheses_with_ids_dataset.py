import editdistance
import numpy as np
import pandas as pd

from utils.logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset


class HypothesesWithIdsDataset(HypothesesDataset):
    def __init__(
            self,
            hypotheses: pd.DataFrame,
            ground_truths: list[str],
            hypotheses_ids: list[list[int]],
            pad_id: int = None,  # If None, no padding and truncation will be applied
            max_seq_length: int = 256,
    ):
        super().__init__(hypotheses, ground_truths)
        self.hypotheses_ids = hypotheses_ids
        self.pad_id = pad_id
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        hypothesis = self._get_hypothesis_text(idx)
        asr_score = self._get_hypothesis_score(idx)
        input_ids = self._get_hypothesis_input_ids(idx)
        input_mask = self._get_hypothesis_input_mask(idx)
        char_length = len(hypothesis)
        ground_truth = self.ground_truths[idx // self.beam_size]
        distance_to_ground_truth = editdistance.eval(hypothesis.split(), ground_truth.split())
        return hypothesis, asr_score, input_ids, input_mask, char_length, distance_to_ground_truth

    def _get_hypothesis_input_ids(self, idx):
        hypothesis_ids = self.hypotheses_ids[idx]

        if self.pad_id is None:
            return np.array(hypothesis_ids)

        if len(hypothesis_ids) > self.max_seq_length:
            Logger.warn(f"Hypothesis at {idx} is longer ({len(hypothesis_ids)}) than max_seq_length ({self.max_seq_length}) and will be truncated.")
        input_ids = [self.pad_id] * self.max_seq_length
        input_ids[: len(hypothesis_ids)] = hypothesis_ids
        return np.array(input_ids)

    def _get_hypothesis_input_mask(self, idx):
        hypothesis_ids = self.hypotheses_ids[idx]
        input_mask = np.zeros(self.max_seq_length)
        input_mask[: len(hypothesis_ids)] = 1
        return input_mask
