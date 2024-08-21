import string
import torch
from transformers import pipeline

from LargeLanguageModel import LargeLanguageModel
from Tokenizer import Tokenizer


class Prompter:
    def __init__(self, llm: LargeLanguageModel, tokenizer: Tokenizer, max_length=10):
        self._generator = pipeline(
            "text-generation",
            model=llm.model,
            tokenizer=tokenizer.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            pad_token_id=tokenizer.pad_id,
            num_return_sequences=1,
            max_length=max_length,
            truncation=True,
        )

    def prompt(self, input_text: str) -> str:
        sequences = self._generator(input_text)
        output_text = sequences[0]["generated_text"]
        return self._sanitize_text(input_text, output_text)

    def _sanitize_text(self, input_text: str, output_text: str):
        text = output_text.replace(input_text, "")
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans('', '', string.digits))
        return text.lower()
