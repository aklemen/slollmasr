import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from best_hypotheses_selector import BestHypothesesSelector
from logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset
from torch_datasets.hypotheses_with_ids_dataset import HypothesesWithIdsDataset


class MaskedRescorer:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 128):
        self.llm_name = llm_name
        self.llm = (
            AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=llm_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.device_to_map_to = "cuda"
        self.batch_size = batch_size

    def run(self, dataset: HypothesesDataset, alpha_weight: float = None, beta_weight: float = None) -> tuple[
        list[float], float, float]:
        hypotheses_ids = [self.tokenizer.encode(h, add_special_tokens=True) for h in dataset.get_hypotheses_texts()]
        with_ids_dataset = HypothesesWithIdsDataset(dataset.hypotheses, dataset.ground_truths, hypotheses_ids, self.tokenizer.pad_token_id)
        data_loader = torch.utils.data.DataLoader(dataset=with_ids_dataset, batch_size=self.batch_size, pin_memory=True)

        with torch.no_grad():
            asr_scores, llm_scores, char_lengths, distances = [], [], [], []
            for batch in tqdm(data_loader):
                _, asr_score, input_ids, input_mask, char_length, distance = batch

                input_ids, input_mask = input_ids.to(self.device_to_map_to), input_mask.to(self.device_to_map_to)
                asr_score = asr_score.to(self.device_to_map_to)
                char_length = char_length.to(self.device_to_map_to)
                distance = distance.to(self.device_to_map_to)

                llm_score = self._calculate_pll_for_batch(input_ids, input_mask)

                asr_scores.append(asr_score)
                llm_scores.append(llm_score)
                char_lengths.append(char_length)
                distances.append(distance)

        asr_scores = torch.cat(asr_scores)
        llm_scores = torch.cat(llm_scores)
        char_lengths = torch.cat(char_lengths).to(asr_scores.dtype)
        distances = torch.cat(distances)

        if alpha_weight is None:
            Logger.info("Alpha weight was not provided. Executing linear search for it...")
            alpha_weight, best_wer = self._find_best_coefficient(with_ids_dataset, asr_scores, llm_scores, distances)
            alpha_weight = np.round(alpha_weight, 3)
            Logger.info(f"alpha_weight={alpha_weight} achieved the best WER ({best_wer}).")

        scores_with_llm = asr_scores + alpha_weight * llm_scores

        if beta_weight is None:
            Logger.info("Beta weight was not provided. Executing linear search for it...")
            beta_weight, best_wer = self._find_best_coefficient(with_ids_dataset, scores_with_llm, char_lengths,
                                                                distances)
            beta_weight = np.round(beta_weight, 3)
            Logger.info(f"beta_weight={beta_weight} achieved the best WER ({best_wer}).")

        new_scores = scores_with_llm + beta_weight * char_lengths

        return new_scores.tolist(), alpha_weight, beta_weight

    def _calculate_pll_for_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.size()
        pll_scores = torch.zeros(batch_size, device=input_ids.device)

        special_tokens = torch.tensor(self.tokenizer.all_special_ids, device=input_ids.device)

        for position in range(seq_len):
            # Skip positions where the attention mask is zero (padding or special tokens)
            if torch.all(attention_mask[:, position] == 0):
                continue

            masked_input = input_ids.clone()
            masked_input[:, position] = self.tokenizer.mask_token_id

            outputs = self.llm(input_ids=masked_input, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch_size, sequence_length, vocab_size]

            orig_token_ids = input_ids[:, position]
            position_log_probs = log_probs[:, position, :]
            orig_token_ids_expanded = orig_token_ids.unsqueeze(1)
            token_log_probs_with_dim = position_log_probs.gather(1, orig_token_ids_expanded)

            token_log_probs = token_log_probs_with_dim.squeeze(1)

            mask_special = ~torch.isin(orig_token_ids, special_tokens)
            mask_attention = attention_mask[:, position].bool()
            mask_valid = mask_special & mask_attention
            token_log_probs = token_log_probs * mask_valid

            pll_scores += token_log_probs

        return pll_scores

    def _find_best_coefficient(self, dataset: HypothesesWithIdsDataset, scores1, scores2, distances):
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
        coefficient_range = [-10, 10]
        coefficient_steps = 10000
        if scores1.isnan().any():
            Logger.warn("scores1 contain NaNs at indices: ", torch.where(scores1.isnan()))
        if scores2.isnan().any():
            Logger.warn("scores2 contain NaNs at indices: ", torch.where(scores2.isnan()))
        scores1_mean = scores1.nanmean().abs().item()
        scores2_mean = scores2.nanmean().abs().item()
        normalization_scale = scores1_mean / scores2_mean
        start = coefficient_range[0] * normalization_scale
        stop = coefficient_range[1] * normalization_scale
        coefficients = np.linspace(start, stop, coefficient_steps)
        return coefficients

    def _calculate_wer(self, distances, ground_truths_length_sum):
        wer = distances.sum() / ground_truths_length_sum
        wer = wer.item()
        return wer
