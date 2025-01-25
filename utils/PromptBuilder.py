from Logger import Logger
from torch_datasets.HypothesesDataset import HypothesesDataset

# from Artur v1.0 (dev)
shot_sample = {
    "hypotheses": [
        "razmere pa me niso dale dihati",
        "razmere pa mi niso dale dihati",
        "razmere pa memi niso dale dihati",
        "razmere pa me niso dale diati",
        "razmere pa mime niso dale dihati",
        "razmere pa mi niso dale diati",
        "razmere pa me niso dale nihhati",
        "razmere me niso dale dihati",
        "razmere pa  niso dale dihati",
        "razmere pa me niso dale dihajati",
    ],
    "transcript": "razmere pa mi niso dale dihati"
}


class PromptBuilder:
    def __init__(self, dataset: HypothesesDataset):
        self.dataset = dataset
        self.num_of_samples = dataset.get_num_of_samples()
        self.beam_size = dataset.get_beam_size()

    def build_for_zero_shot_ger(self):
        pre_prompt = (
            f"Izvedi popravljanje napak na najboljših {self.beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR, so naslednje:"
        )
        post_prompt = "Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        prompts = []
        for sample_idx in range(self.num_of_samples):
            hypotheses = self._get_stringified_hypotheses_for_sample(sample_idx, "hipoteza")
            prompt = f"{pre_prompt}\n\n{hypotheses}\n\n{post_prompt}"
            prompts.append(prompt)

        Logger.info(f"{len(prompts)} prompts built.")
        return prompts

    def build_for_one_shot_ger(self):
        shot_hypotheses = self._stringify_hypotheses(shot_sample["hypotheses"][:self.beam_size], 'hipoteza')
        shot_transcript = shot_sample["transcript"]

        pre_prompt = (
            f"Izvedi popravljanje napak na najboljših {self.beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
            f"Hipoteze so navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR. "
            f"Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed. "
            f"Tukaj je primer naloge:\n\n"
            f"{shot_hypotheses}\n\n"
            f"Tvoj izhod: {shot_transcript}\n\n"
            f"Prosim, zgleduj se po zgornjem primeru. Prosim, začni:"
        )

        prompts = []
        for sample_idx in range(self.num_of_samples):
            hypotheses_list = self._get_stringified_hypotheses_for_sample(sample_idx, "hipoteza")
            prompt = f"{pre_prompt}\n\n{hypotheses_list}"
            prompts.append(prompt)

        Logger.info(f"{len(prompts)} prompts built.")
        return prompts

    def _get_stringified_hypotheses_for_sample(self, sample_idx: int, tag_name: str) -> str:
        hypotheses_list = self._get_hypotheses_for_sample(sample_idx)
        return self._stringify_hypotheses(hypotheses_list, tag_name)

    def _get_hypotheses_for_sample(self, sample_idx: int) -> list[str]:
        from_dataset_idx = sample_idx * self.beam_size
        to_dataset_idx = (sample_idx + 1) * self.beam_size
        return [self.dataset[i][0] for i in range(from_dataset_idx, to_dataset_idx)]

    def _stringify_hypotheses(self, hypotheses: list[str], tag_name: str) -> str:
        hypotheses_with_tags = [
            f"<{tag_name}{idx}> {hypothesis} </{tag_name}{idx}>" for idx, hypothesis in enumerate(hypotheses)
        ]
        return "\n".join(hypotheses_with_tags)
