from utils.Prompter import Prompter
from torch_datasets.HypothesesDataset import HypothesesDataset
from utils.PromptBuilder import PromptBuilder


class OneShotGer(Prompter):
    shot = {
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
        "ground_truth": "razmere pa mi niso dale dihati"
    }

    def run(self, dataset: HypothesesDataset):
        prompts = self._build_prompts(dataset)
        return self.execute_prompts(prompts)

    def _build_prompts(self, dataset: HypothesesDataset) -> list[str]:
        prompt_builder = PromptBuilder(dataset)
        beam_size = dataset.get_beam_size()

        # from Artur v1.0 (dev)
        shot_hypotheses = prompt_builder.stringify_hypotheses(self.shot["hypotheses"][:beam_size], 'hipoteza')
        shot_transcript = self.shot["ground_truth"]

        prefix = (
            f"Izvedi popravljanje napak na najboljših {beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
            f"Hipoteze so navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR. "
            f"Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed. "
            f"Tukaj je primer naloge:\n\n"
            f"{shot_hypotheses}\n\n"
            f"Tvoj izhod: {shot_transcript}\n\n"
            f"Prosim, zgleduj se po zgornjem primeru. Prosim, začni:"
        )

        return prompt_builder.build(prefix, "hipoteza")