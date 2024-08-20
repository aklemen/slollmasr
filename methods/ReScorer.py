import inspect
import logging
import torch
from tqdm import tqdm

from HypothesesDataset import HypothesesDataset
from llms.LargeLanguageModel import LargeLanguageModel


class ReScorer:
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
                ground_truths, hypotheses, asr_scores, llm_scores = [], [], [], []
                for batch in tqdm(data_loader):
                    ground_truth, hypothesis, asr_score, input_ids, input_mask = batch

                    max_len_in_batch = input_mask.sum(dim=0).argmin().item()
                    input_ids, input_mask = input_ids[:, :max_len_in_batch], input_mask[:, :max_len_in_batch]
                    input_ids, input_mask = input_ids.to(self.device_to_map_to), input_mask.to(self.device_to_map_to)
                    asr_score = asr_score.to(self.device_to_map_to)

                    if support_attention_mask:
                        output = self.llm.model(input_ids=input_ids, attention_mask=input_mask)
                    else:
                        output = self.llm.model(input_ids=input_ids)

                    log_probs = torch.nn.functional.log_softmax(output.logits, dim=-1)

                    target_log_probs = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
                    llm_score = torch.sum(target_log_probs * input_mask[:, 1:], dim=-1)

                    ground_truths.append(ground_truth)
                    hypotheses.append(hypothesis)
                    asr_scores.append(asr_score)
                    llm_scores.append(llm_score)

        asr_scores = torch.cat(asr_scores).view(-1, beam_size)
        llm_scores = torch.cat(llm_scores).view(-1, beam_size)

        new_scores = asr_scores + alpha_weight * llm_scores

        new_scores_flatten = new_scores.flatten()
        return new_scores_flatten.tolist()
