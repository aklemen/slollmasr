import string
import torch
from tqdm import tqdm
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
            max_new_tokens=512,
            return_full_text=False,
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

        new_scores = []
        for  sample_idx in range(dataset.get_num_of_samples()):
            prompt = self._generate_prompt(dataset, sample_idx)
            sequences = self._generator(prompt)
            result = sequences[-1]["generated_text"]
            partial_new_scores = ','.split(result)
            if len(partial_new_scores) != beam_size:
                raise Exception(f"Expected {beam_size} scores, but got list of length {len(partial_new_scores)}. Result from LLM: {result}")
            partial_new_scores = [float(score) for score in partial_new_scores]
            new_scores.append(partial_new_scores)

        return [item for sublist in new_scores for item in sublist]

    def _generate_prompt(self, dataset, sample_idx):
        prompt = ("You will be given a list of hypothesis from the ASR system, together with their respective scores. "
                  "The format of the list is the following: <hypothesis></hypothesis><score></score>,<hypothesis></hypothesis><score></score>,... "
                  "Your task is to rescore the hypotheses and output only the new scores, separated by a comma.\n"
                  "Example: 1.23,3.24,0.12,4.56,2.34\n"
                  "Do not include any other information in the output. "
                  "The list of hypotheses is the following:\n")
        from_idx = sample_idx * dataset.get_beam_size()
        to_idx = (sample_idx + 1) * dataset.get_beam_size()
        for i in range(from_idx, to_idx):
            prompt += f'<hypothesis>{dataset[i][0]}</hypothesis><score>{str(dataset[i][1])}</score>\n'
        return prompt