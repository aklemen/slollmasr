import inspect
import logging
import torch
from tqdm import tqdm

from torch_datasets.HypothesesDataset import HypothesesDataset
from LargeLanguageModel import LargeLanguageModel


class CausalReScorer:
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm
        self.device_to_map_to = "cuda"
        self.batch_size = 128

    def re_score(self, dataset: HypothesesDataset, alpha_weight: int = 0.5) -> list[float]:
        beam_size = dataset.get_beam_size()
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size)

        if "attention_mask" in inspect.getfullargspec(self.llm.model.forward).args:
            logging.info(f'Attention mask is supported by "{self.llm.name}" and will be used.')
            support_attention_mask = True
        else:
            logging.info(f'Attention mask NOT supported by "{self.llm.name}" and will NOT be used.')
            support_attention_mask = False

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                hypotheses, asr_scores, llm_scores = [], [], []
                for batch in tqdm(data_loader):
                    hypothesis, asr_score, input_ids, input_mask = batch

                    max_len_in_batch = input_mask.sum(dim=0).argmin().item()
                    input_ids, input_mask = input_ids[:, :max_len_in_batch], input_mask[:, :max_len_in_batch]
                    input_ids, input_mask = input_ids.to(self.device_to_map_to), input_mask.to(self.device_to_map_to)
                    asr_score = asr_score.to(self.device_to_map_to)

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

                    shifted_input_mask = input_mask[:, 1:]
                    llm_score = torch.sum(target_log_probs * shifted_input_mask, dim=-1)

                    hypotheses.append(hypothesis)
                    asr_scores.append(asr_score)
                    llm_scores.append(llm_score)

        asr_scores = torch.cat(asr_scores)
        llm_scores = torch.cat(llm_scores)

        new_scores = asr_scores + alpha_weight * llm_scores

        return new_scores.tolist()
