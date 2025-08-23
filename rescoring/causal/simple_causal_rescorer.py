import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset


class SimpleCausalReScorer:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 128):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding_side="right")
        if self._tokenizer.pad_token is None:
            Logger.info(f"No pad_token available. Setting pad_token to eos_token: {self._tokenizer.eos_token}")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._llm = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=llm_name,
                is_decoder=True,
                attn_implementation="flash_attention_2",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            .eval()
        )
        self._batch_size = batch_size

    def run(self, dataset: HypothesesDataset, alpha_weight: float, beta_weight: float) -> tuple[list[float], float, float]:
        assert isinstance(alpha_weight, float), f"alpha_weight should be a float, but got {type(alpha_weight)}"
        assert isinstance(beta_weight, float), f"beta_weight should be a float, but got {type(beta_weight)}"

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self._batch_size, pin_memory=True)

        asr_scores, llm_scores, char_lengths = [], [], []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                hypotheses_batch = batch[0]
                asr_scores_batch = batch[1]
                char_lengths_batch = [len(hypothesis) for hypothesis in hypotheses_batch]

                inputs = self._tokenizer(hypotheses_batch, return_tensors="pt", padding=True, add_special_tokens=True)
                output = self._llm(**inputs)

                log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)  # [batch_size, sequence_length, vocab_size]

                # We don't care about the last probabilities, because they are for the next token (outside our sequence)
                log_probs_without_last = log_probs[:, :-1]  # [batch_size, sequence_length - 1, vocab_size]

                # We shift the IDs to align them with log probabilities (so probabilities at position 0 are for token at position 0)
                shifted_input_ids = inputs["input_ids"][:, 1:]  # [batch_size, sequence_length - 1]

                # We need to only get the probabilities of our input tokens (not across the whole vocabulary)
                target_log_probs = log_probs_without_last.gather(2, shifted_input_ids.unsqueeze(2)).squeeze(2)

                shifted_input_mask = inputs["attention_mask"][:, 1:] # [batch_size, sequence_length - 1]
                llm_scores_batch = torch.sum(target_log_probs * shifted_input_mask, dim=-1)

                asr_scores.extend(asr_scores_batch)
                llm_scores.extend(llm_scores_batch)
                char_lengths.extend(char_lengths_batch)

        asr_scores = torch.tensor(asr_scores)
        llm_scores = torch.tensor(llm_scores)
        char_lengths = torch.tensor(char_lengths)

        scores_with_llm = asr_scores + alpha_weight * llm_scores
        new_scores = scores_with_llm + beta_weight * char_lengths
        return new_scores.tolist(), alpha_weight, beta_weight