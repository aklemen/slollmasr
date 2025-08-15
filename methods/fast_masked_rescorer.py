import numpy as np
import torch
from numba.cuda.libdevice import llmax
from torch.nn.functional import batch_norm
from torch.xpu import device
from transformers import AutoModelForMaskedLM, AutoTokenizer

from torch_datasets.hypotheses_dataset import HypothesesDataset

"""
Limitation: Works only on one GPU.
"""
class FastMaskedRescorer:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 1024):
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model = AutoModelForMaskedLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.batch_size = batch_size

    def run(self, dataset: HypothesesDataset, alpha_weight: float = None, beta_weight: float = None) -> tuple[list[float], float, float]:
        hypotheses_tokens = [self.tokenizer.tokenize(h, add_special_tokens=False) for h in dataset.get_hypotheses_texts()]

        masked_sequences, hypothesis_indices, target_ids = self._prepare_masked_sequences(hypotheses_tokens)

        llm_scores = torch.zeros(len(masked_sequences), device=self.device, dtype=torch.float16)

        for i in range(0, len(masked_sequences), self.batch_size):
            batch_masked_sentences = masked_sequences[i:i + self.batch_size]
            batch_inputs = self.tokenizer(batch_masked_sentences, return_tensors="pt", padding=True).to(self.device)
            batch_target_ids = target_ids[i:i + self.batch_size]

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits

            mask_positions = (batch_inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            indices_of_sentences_with_mask = mask_positions[0]
            positions_of_masks_in_sentences = mask_positions[1]

            batch_scores = logits[indices_of_sentences_with_mask, positions_of_masks_in_sentences, batch_target_ids]
            llm_scores[i:i + len(batch_scores)] = batch_scores

        hypothesis_scores_gpu = torch.zeros(len(hypotheses_tokens), device=self.device, dtype=torch.float16)
        hypothesis_scores_gpu.scatter_add_(0, hypothesis_indices, llm_scores)

        asr_scores = torch.tensor(dataset.get_hypotheses_scores(), device=self.device, dtype=torch.float16)
        char_lengths = torch.tensor([len(h) for h in dataset.get_hypotheses_texts()], device=self.device, dtype=torch.int16)

        scores_with_llm = asr_scores + 0.5 * hypothesis_scores_gpu
        new_scores = scores_with_llm + 0.5 * char_lengths

        return new_scores.tolist(), alpha_weight, beta_weight

    """
    all_masked_sentences: List of sentences with one token masked at a time.
    hypothesis_indices: List of indices corresponding to the original hypotheses for each masked sentence
    target_ids: List of token IDs corresponding to the masked tokens.
    """
    def _prepare_masked_sequences(self, hypotheses_tokens):
        all_masked_sentences = []
        all_target_tokens = []
        hypothesis_indices = []

        mask_token = self.tokenizer.mask_token

        for hypothesis_index, hypothesis_tokens in enumerate(hypotheses_tokens):
            for i, token in enumerate(hypothesis_tokens):
                masked_tokens = [mask_token if j == i else t for j, t in enumerate(hypothesis_tokens)]
                masked_sentence = self.tokenizer.convert_tokens_to_string(masked_tokens)

                all_masked_sentences.append(masked_sentence)
                all_target_tokens.append(token)
                hypothesis_indices.append(hypothesis_index)

        target_ids = self.tokenizer.convert_tokens_to_ids(all_target_tokens)

        hypothesis_indices_gpu = torch.tensor(hypothesis_indices, device=self.device, dtype=torch.int32)
        target_ids_gpu = torch.tensor(target_ids, device=self.device, dtype=torch.int32)

        return all_masked_sentences, hypothesis_indices_gpu, target_ids_gpu
