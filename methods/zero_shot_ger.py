from utils.prompter import Prompter
from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.prompt_builder import PromptBuilder


class ZeroShotGer(Prompter):
    def run(self, dataset: HypothesesDataset):
        prompts = self._build_prompts(dataset)
        return self.execute_prompts(prompts)

    def _build_prompts(self, dataset: HypothesesDataset) -> list[str]:
        prompt_builder = PromptBuilder(dataset)
        prefix = (
            f"Izvedi popravljanje napak na najboljših {dataset.get_beam_size()} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (ASR). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost iz sistema ASR, so naslednje:"
        )
        postfix = "Prosim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        return prompt_builder.build(prefix, "hipoteza", postfix)