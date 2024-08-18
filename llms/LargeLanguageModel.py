from transformers import AutoModelForCausalLM, AutoTokenizer


class LargeLanguageModel:
    def __init__(self, name: str, device="cuda"):
        self.name = name
        self.device = device
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=name,
                is_decoder=True,
            )
            .to(device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=name,
            use_fast=True,
        )
