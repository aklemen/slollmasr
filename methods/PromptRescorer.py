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

        # if tokenizer.are_chat_templates_supported():
        #     self.prompt = {
        #         "role": "user",
        #         "content": prompt,
        #     }
        # else:
        #     self.prompt = prompt

    def run(self, dataset: HypothesesDataset) -> list[float]:
        beam_size = dataset.get_beam_size()
        hypotheses_texts = dataset.get_hypotheses_texts()
        hypotheses_groups = [hypotheses_texts[i:i + beam_size] for i in range(0, len(hypotheses_texts), beam_size)]

        new_scores = []
        for  (i, group) in enumerate(hypotheses_groups):
            prompt = self._generate_prompt(group)
            sequences = self._generator(prompt)
            result = sequences[-1]["generated_text"]
            partial_new_scores = ','.split(result)
            if len(partial_new_scores) != beam_size:
                raise Exception(f"Expected {beam_size} scores, but got list of length {len(partial_new_scores)}. Result from LLM: {result}")
            partial_new_scores = [float(score) for score in partial_new_scores]
            new_scores.append(partial_new_scores)

        return [item for sublist in new_scores for item in sublist]

    def _sanitize_text(self, input_text: str, output_text: str):
        text = output_text.replace(input_text, "")
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()

    def _generate_prompt(self, hypotheses: list[str]):
        prompt = ("You will be given a list of hypothesis from the ASR system, together with their respective scores. "
                  "Your task is to rescore the hypotheses and output only the new score, separated by a coma.\n"
                  "Example: 1.23,3.24,0.12,4.56,2.34\n"
                  "Do not include any other information in the output."
                  "The list of hypotheses is the following:\n")
        return prompt + '\n'.join(hypotheses)
