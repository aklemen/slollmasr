from torch_datasets.hypotheses_dataset import HypothesesDataset
from prompting.build_chats import build_prompt_dataset
from data.build_h2t_prompt import build_h2t_prompt
from prompting.prompter import Prompter


class H2TMapping:
    def __init__(self, llm_name_or_path: str, tokenizer_name_or_path: str, batch_size: int = 8):
        self.prompter = Prompter(llm_name_or_path, tokenizer_name_or_path, batch_size)

    def run(self, dataset: HypothesesDataset):
        chats = build_prompt_dataset(dataset, build_h2t_prompt)
        return self.prompter.execute_prompts(chats)
