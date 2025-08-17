import torch
from tqdm import tqdm
import editdistance
from transformers import AutoModelForMaskedLM, AutoTokenizer

from logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.coefficient_finder import CoefficientFinder

"""
Limitation: Works only on one GPU.
"""
class FastMaskedRescorer:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 1024):
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model = AutoModelForMaskedLM.from_pretrained(llm_name, device_map=self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.batch_size = batch_size
        self.coefficient_finder = CoefficientFinder(self.device)

    def run(self, dataset: HypothesesDataset, alpha_weight: float = None, beta_weight: float = None) -> tuple[list[float], float, float]:
        hypotheses_tokens = [self.tokenizer.tokenize(h, add_special_tokens=False) for h in dataset.get_hypotheses_texts()]

        Logger.info("Preparing masked sequences for LLM scoring ...")
        masked_sequences, hypothesis_indices, target_ids = self._prepare_masked_sequences(hypotheses_tokens)

        per_mask_llm_scores = torch.zeros(len(masked_sequences), device=self.device, dtype=torch.float32)

        Logger.info("Scoring hypotheses with LLM ...")
        for i in tqdm(range(0, len(masked_sequences), self.batch_size)):
            batch_masked_sentences = masked_sequences[i:i + self.batch_size]
            batch_inputs = self.tokenizer(batch_masked_sentences, return_tensors="pt", padding=True).to(self.device)
            batch_target_ids = target_ids[i:i + self.batch_size]

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits

            mask_positions = (batch_inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            indices_of_sentences_with_mask = mask_positions[0]
            positions_of_masks_in_sentences = mask_positions[1]

            mask_logits = logits[indices_of_sentences_with_mask, positions_of_masks_in_sentences, :]

            log_probs = torch.nn.functional.log_softmax(mask_logits, dim=-1)
            batch_scores = log_probs.gather(1, batch_target_ids.unsqueeze(1)).squeeze(1)

            per_mask_llm_scores[i:i + batch_scores.size(0)] = batch_scores.to(per_mask_llm_scores.dtype)

        llm_scores = torch.zeros(len(hypotheses_tokens), device=self.device, dtype=torch.float32)
        llm_scores.scatter_add_(0, hypothesis_indices, per_mask_llm_scores)
        asr_scores = torch.tensor(dataset.get_hypotheses_scores(), device=self.device, dtype=torch.float32)

        if alpha_weight is None or beta_weight is None:
            Logger.info("Calculating distances for WER calculation ...")
            ground_truths = dataset.get_ground_truths()
            distances = []
            for i, hypothesis in enumerate(tqdm(dataset.get_hypotheses_texts())):
                ground_truth = ground_truths[i // dataset.get_beam_size()]
                distance = editdistance.eval(hypothesis.split(), ground_truth.split())
                distances.append(distance)
            distances = torch.tensor(distances, device=self.device, dtype=torch.float32)

        Logger.info("Calculating character lengths for beta weight search ...")
        char_lengths = torch.tensor([len(h) for h in dataset.get_hypotheses_texts()], device=self.device, dtype=torch.float32)

        if alpha_weight is None:
            Logger.info("Alpha weight was not provided. Executing linear search for it...")
            alpha_weight, best_wer = self.coefficient_finder.find_best_coefficient(dataset, asr_scores, llm_scores, distances)
            Logger.info(f"alpha_weight={alpha_weight} achieved the best WER ({best_wer}).")

        scores_with_llm = asr_scores + alpha_weight * llm_scores

        if beta_weight is None:
            Logger.info("Beta weight was not provided. Executing linear search for it...")
            beta_weight, best_wer = self.coefficient_finder.find_best_coefficient(dataset, scores_with_llm, char_lengths, distances)
            Logger.info(f"beta_weight={beta_weight} achieved the best WER ({best_wer}).")

        new_scores = scores_with_llm + beta_weight * char_lengths

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

        for hypothesis_index, hypothesis_tokens in enumerate(tqdm(hypotheses_tokens)):
            for i, token in enumerate(hypothesis_tokens):
                masked_tokens = [mask_token if j == i else t for j, t in enumerate(hypothesis_tokens)]
                masked_sentence = self.tokenizer.convert_tokens_to_string(masked_tokens)

                all_masked_sentences.append(masked_sentence)
                all_target_tokens.append(token)
                hypothesis_indices.append(hypothesis_index)

        target_ids = self.tokenizer.convert_tokens_to_ids(all_target_tokens)

        hypothesis_indices_gpu = torch.tensor(hypothesis_indices, device=self.device, dtype=torch.long)
        target_ids_gpu = torch.tensor(target_ids, device=self.device, dtype=torch.long)

        return all_masked_sentences, hypothesis_indices_gpu, target_ids_gpu
