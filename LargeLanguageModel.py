from transformers import AutoModelForCausalLM


class LargeLanguageModel:
    def __init__(self, name: str):
        self.name = name
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=name,
                is_decoder=True,
                device_map="auto",
                torch_dtype="auto",
                temperature=0 # To get as deterministic results as possible - see https://arxiv.org/html/2408.04667v1
            )
            .eval()
        )
