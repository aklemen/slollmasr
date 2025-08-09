from peft.helpers import check_if_peft_model

from torch_datasets.hypotheses_dataset import HypothesesDataset
from utils.build_chats import build_prompt_dataset
from utils.build_h2t_prompt import build_h2t_prompt
from utils.prompter import Prompter


class H2TMapping:
    def __init__(self, llm_name_or_path: str, tokenizer_name_or_path: str, batch_size: int = 8):
        if not check_if_peft_model(llm_name_or_path):
            raise ValueError(f"The provided model '{llm_name_or_path}' is not a PEFT model. "
                             "'h2t-mapping' method only allows using PEFT models.")
        self.prompter = Prompter(llm_name_or_path, tokenizer_name_or_path, batch_size)

    def run(self, dataset: HypothesesDataset):
        chats = build_prompt_dataset(dataset, build_h2t_prompt)
        return self.prompter.execute_prompts(chats)
