from abc import ABC

from transformers import AutoModelForCausalLM, AutoTokenizer


class ReEvaluator(ABC):
    def __init__(self, llm_name: str):
        self.device = "cuda"
        self.llm_name = llm_name
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=llm_name,
                is_decoder=True,
            )
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=llm_name,
            use_fast=True,
        )
