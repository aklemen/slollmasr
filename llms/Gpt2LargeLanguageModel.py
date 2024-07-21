from transformers import pipeline
import torch

from llms.LargeLanguageModel import LargeLanguageModel


class Gpt2LargeLanguageModel(LargeLanguageModel):
    def __init__(self):
        model_name = "gpt2"
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def prompt(self, input_text: str) -> str:
        sequences = self.generator(
            input_text,
            num_return_sequences=1,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )

        first_sequence = sequences[0]["generated_text"]
        return first_sequence.replace(input_text, "")

    def score(self, input_text: str) -> float:
        pass
