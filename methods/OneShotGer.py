from utils.Prompter import Prompter
from torch_datasets.HypothesesDataset import HypothesesDataset
from utils.PromptBuilder import PromptBuilder


class OneShotGer(Prompter):
    def run(self, dataset: HypothesesDataset):
        prompt_builder = PromptBuilder(dataset)
        prompts = prompt_builder.build_for_one_shot_ger()
        return self.execute_prompts(prompts)