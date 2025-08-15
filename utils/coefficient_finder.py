import numpy as np
import torch
from tqdm import tqdm

from best_hypotheses_selector import BestHypothesesSelector
from logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset


class CoefficientFinder:
    def __init__(self, device_to_map_to: str):
        self.device_to_map_to = device_to_map_to

    def find_best_coefficient(self, dataset: HypothesesDataset, scores1, scores2, distances) -> tuple[float, float]:
        ground_truths_word_lengths_sum = sum(len(ground_truth.split()) for ground_truth in dataset.get_ground_truths())
        ground_truths_word_lengths_sum = torch.tensor(ground_truths_word_lengths_sum).to(self.device_to_map_to)
        coefficients = self._get_coefficients(scores1, scores2)
        best_coefficient = coefficients[0]
        best_wer = 10000
        for coefficient in tqdm(coefficients):
            new_scores = scores1 + coefficient * scores2
            _, _, new_best_indices = BestHypothesesSelector.select(dataset, new_scores.tolist())
            new_best_indices = torch.tensor(new_best_indices).to(self.device_to_map_to)
            best_hypothesis_distances = distances.gather(dim=0, index=new_best_indices)
            wer = self._calculate_wer(best_hypothesis_distances, ground_truths_word_lengths_sum)
            if wer < best_wer:
                best_coefficient = coefficient
                best_wer = wer
        best_coefficient = np.round(best_coefficient, 3)
        return best_coefficient, best_wer

    @staticmethod
    def _get_coefficients(scores1, scores2) -> list[float]:
        coefficient_range = [-10, 10]
        coefficient_steps = 10000
        if scores1.isnan().any():
            Logger.warn(f"scores1 contain NaNs at indices: {torch.where(scores1.isnan())}")
        if scores2.isnan().any():
            Logger.warn(f"scores2 contain NaNs at indices: {torch.where(scores2.isnan())}")
        scores1_mean = scores1.nanmean().abs().item()
        scores2_mean = scores2.nanmean().abs().item()
        normalization_scale = scores1_mean / scores2_mean
        start = coefficient_range[0] * normalization_scale
        stop = coefficient_range[1] * normalization_scale
        coefficients = np.linspace(start, stop, coefficient_steps)
        return coefficients.tolist()

    @staticmethod
    def _calculate_wer(distances, ground_truths_length_sum) -> float:
        wer = distances.sum() / ground_truths_length_sum
        wer = wer.item()
        return wer
