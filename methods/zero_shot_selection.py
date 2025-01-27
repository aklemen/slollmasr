from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.prompt_builder import PromptBuilder
from utils.prompter import Prompter


class ZeroShotSelection(Prompter):
    def run(self, dataset: HypothesesDataset):
        prompts = self._build_prompts(dataset)
        return self.execute_prompts(prompts)

    def _build_prompts(self, dataset: HypothesesDataset) -> list[str]:
        prompt_builder = PromptBuilder(dataset)

        prefix = (
            f"Kot jezikovni model izvedi ponovno ocenjevanje najboljših {dataset.get_beam_size()} izhodov, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR, so naslednje:"
        )
        postfix = "Prosim, izpiši le izbrano najboljšo hipotezo iz sistema ASR, brez dodatnih razlag ali besed."

        return prompt_builder.build(prefix, "hipoteza", postfix)
