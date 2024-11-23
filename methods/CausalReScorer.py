import inspect

import numpy as np
import torch
from tqdm import tqdm

from BestHypothesesSelector import BestHypothesesSelector
from torch_datasets.HypothesesDataset import HypothesesDataset
from LargeLanguageModel import LargeLanguageModel


class CausalReScorer:
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm
        self.device_to_map_to = "cuda"
        self.batch_size = 128

    def re_score(self, dataset: HypothesesDataset, alpha_weight: int = None, beta_weight: int = None) -> list[float]:
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size)

        if "attention_mask" in inspect.getfullargspec(self.llm.model.forward).args:
            print(f'Attention mask is supported by "{self.llm.name}" and will be used.')
            support_attention_mask = True
        else:
            print(f'Attention mask NOT supported by "{self.llm.name}" and will NOT be used.')
            support_attention_mask = False

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                hypotheses, asr_scores, llm_scores, char_lengths, distances = [], [], [], [], []
                for batch in tqdm(data_loader):
                    _, asr_score, input_ids, input_mask, char_length, distance = batch

                    max_len_in_batch = input_mask.sum(dim=0).argmin().item()
                    input_ids, input_mask = input_ids[:, :max_len_in_batch], input_mask[:, :max_len_in_batch]
                    input_ids, input_mask = input_ids.to(self.device_to_map_to), input_mask.to(self.device_to_map_to)
                    asr_score = asr_score.to(self.device_to_map_to)
                    char_length = char_length.to(self.device_to_map_to)
                    distance = distance.to(self.device_to_map_to)

                    if support_attention_mask:
                        output = self.llm.model(input_ids=input_ids, attention_mask=input_mask)
                    else:
                        output = self.llm.model(input_ids=input_ids)

                    log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)  # [batch_size, sequence_length, vocab_size]

                    # We don't care about the last probabilities, because they are for the next token (outside our sequence)
                    log_probs_without_last = log_probs[:, :-1]  # [batch_size, sequence_length - 1, vocab_size]

                    # We shift the IDs to align them with log probabilities (so probabilities at position 0 are for token at position 0)
                    shifted_input_ids = input_ids[:, 1:]  # [batch_size, sequence_length - 1]

                    # We need to only get the probabilities of our input tokens (not across the whole vocabulary)
                    target_log_probs = log_probs_without_last.gather(2, shifted_input_ids.unsqueeze(2)).squeeze(2)

                    shifted_input_mask = input_mask[:, 1:] # [batch_size, sequence_length - 1]
                    llm_score = torch.sum(target_log_probs * shifted_input_mask, dim=-1)

                    asr_scores.append(asr_score)
                    llm_scores.append(llm_score)
                    char_lengths.append(char_length)
                    distances.append(distance)

        asr_scores = torch.cat(asr_scores)
        llm_scores = torch.cat(llm_scores)
        char_lengths = torch.cat(char_lengths).to(asr_scores.dtype)
        distances = torch.cat(distances)

        if alpha_weight is None:
            print("Alpha weight was not provided. Executing linear search for it...")
            alpha_weight, best_wer = self._find_best_coefficient(dataset, asr_scores, llm_scores, distances)
            alpha_weight = np.round(alpha_weight, 3)
            print(f"alpha_weight={alpha_weight} achieved the best WER ({best_wer}).")

        scores_with_llm = asr_scores + alpha_weight * llm_scores

        if beta_weight is None:
            print("Beta weight was not provided. Executing linear search for it...")
            beta_weight, best_wer = self._find_best_coefficient(dataset, scores_with_llm, char_lengths, distances)
            beta_weight = np.round(beta_weight, 3)
            print(f"beta_weight={beta_weight} achieved the best WER ({best_wer}).")

        new_scores = scores_with_llm + beta_weight * char_lengths

        return new_scores.tolist()

    def _find_best_coefficient(self, dataset: HypothesesDataset, scores1, scores2, distances):
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
        return best_coefficient, best_wer

    def _get_coefficients(self, scores1, scores2):
        coefficient_range = [0, 10]
        coefficient_steps = 10000
        scores1_mean = scores1.mean().abs().item()
        scores2_mean = scores2.mean().abs().item()
        normalization_scale = scores1_mean / scores2_mean
        start = coefficient_range[0] * normalization_scale
        stop = coefficient_range[1] * normalization_scale
        coefficients = np.linspace(start, stop, coefficient_steps)
        return coefficients

    def _calculate_wer(self, distances, ground_truths_length_sum):
        wer = distances.sum() / ground_truths_length_sum
        wer = wer.item()
        return wer

