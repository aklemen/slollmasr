import string
import torch
from transformers import pipeline

from LargeLanguageModel import LargeLanguageModel
from Tokenizer import Tokenizer
from methods.Method import Method
from torch_datasets.HypothesesDataset import HypothesesDataset


class PromptRescorer(Method):
    def __init__(self, llm: LargeLanguageModel, tokenizer: Tokenizer):
        super().__init__(llm, tokenizer)
        self._generator = pipeline(
            "text-generation",
            model=llm.model,
            tokenizer=tokenizer.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            num_return_sequences=1,
        )

    def run(self, dataset: HypothesesDataset) -> list[float]:
        # sequences = self._generator(input_text)
        # output_text = sequences[0]["generated_text"]
        # return self._sanitize_text(input_text, output_text)
        # TODO
        return dataset.get_hypotheses_scores()

    def _sanitize_text(self, input_text: str, output_text: str):
        text = output_text.replace(input_text, "")
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()
