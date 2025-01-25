from Logger import Logger
from torch_datasets.HypothesesDataset import HypothesesDataset


class PromptBuilder:
    def __init__(self, dataset: HypothesesDataset):
        self.dataset = dataset
        self.num_of_samples = dataset.get_num_of_samples()
        self.beam_size = dataset.get_beam_size()

    def build_for_zero_shot_ger(self):
        pre_prompt = (
            f"Izvedi popravljanje napak na najboljših {self.beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (Automatic Speech Recognition). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost sistema ASR, so naslednje:"
        )
        post_prompt = "Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        prompts = []
        for sample_idx in range(self.num_of_samples):
            hypotheses_list = self._generate_hypotheses_list(sample_idx)
            prompt = f"{pre_prompt}\n\n{hypotheses_list}\n\n{post_prompt}"
            prompts.append(prompt)

        Logger.info(f"{len(prompts)} prompts built.")
        return prompts

    def _generate_hypotheses_list(self, sample_idx):
        from_dataset_idx = sample_idx * self.beam_size
        to_dataset_idx = (sample_idx + 1) * self.beam_size
        hypotheses_list = [
            f'<hipoteza{i + 1}> {self.dataset[dataset_idx][0]} </hipoteza{i + 1}>'
            for i, dataset_idx in enumerate(range(from_dataset_idx, to_dataset_idx))
        ]
        return "\n".join(hypotheses_list)