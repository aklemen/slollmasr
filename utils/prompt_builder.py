from logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset

class PromptBuilder:
    def __init__(self, dataset: HypothesesDataset):
        self.dataset = dataset
        self.num_of_samples = dataset.get_num_of_samples()
        self.beam_size = dataset.get_beam_size()

    def build(self, prefix: str, hypothesis_list_tag: str, postfix: str = None) -> list[str]:
        postfix = "" if postfix is None else f"\n\n{postfix}"
        prompts = []
        for sample_idx in range(self.dataset.get_num_of_samples()):
            hypotheses = self._get_stringified_hypotheses_for_sample(sample_idx, hypothesis_list_tag)
            prompt = f"{prefix}\n\n{hypotheses}{postfix}"
            prompts.append(prompt)

        Logger.info(f"{len(prompts)} prompts built.")

        return prompts

    def stringify_hypotheses(self, hypotheses: list[str], hypothesis_list_tag: str) -> str:
        hypotheses_with_tags = [
            f"<{hypothesis_list_tag}{idx}> {hypothesis} </{hypothesis_list_tag}{idx}>" for idx, hypothesis in enumerate(hypotheses)
        ]
        return "\n".join(hypotheses_with_tags)

    def _get_stringified_hypotheses_for_sample(self, sample_idx: int, hypothesis_list_tag: str) -> str:
        hypotheses_list = self._get_hypotheses_for_sample(sample_idx)
        return self.stringify_hypotheses(hypotheses_list, hypothesis_list_tag)

    def _get_hypotheses_for_sample(self, sample_idx: int) -> list[str]:
        from_dataset_idx = sample_idx * self.beam_size
        to_dataset_idx = (sample_idx + 1) * self.beam_size
        return [self.dataset[i][0] for i in range(from_dataset_idx, to_dataset_idx)]
