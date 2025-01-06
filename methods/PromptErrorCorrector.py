import string
from typing import Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

from LargeLanguageModel import LargeLanguageModel
from Tokenizer import Tokenizer
from methods.Method import Method
from torch_datasets.HypothesesDataset import HypothesesDataset


class PromptsDataset(Dataset):
    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __len__(self):
        return len(self.prompts)


class TokenizedPromptsDataset(Dataset):
    def __init__(self, prompts_ids):
        self.prompts_ids = prompts_ids

    def __getitem__(self, idx):
        return self.prompts_ids[idx]

    def __len__(self):
        return len(self.prompts_ids)


class PromptErrorCorrector(Method):
    def __init__(self, llm: LargeLanguageModel, tokenizer: Tokenizer):
        super().__init__(llm, tokenizer)
        self.llm = llm
        self.tokenizer = tokenizer
        self.device_to_map_to = "cuda"
        self.batch_size = 128

    def run(self, dataset: HypothesesDataset) -> list[str]:
        prompts = self._build_prompts(dataset)
        print(f"{len(prompts)} prompts built. Tokenizing ...")
        model_inputs = self.tokenizer.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        print(f"Prompts tokenized. Number of model_inputs: {len(model_inputs)}.")
        tokenized_prompts_dataset = TokenizedPromptsDataset(model_inputs)
        data_loader = torch.utils.data.DataLoader(dataset=tokenized_prompts_dataset, batch_size=self.batch_size)
        print("Data loader created.")

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                best_hypotheses = []
                for batch in tqdm(data_loader):
                    print("Starting batch ...")
                    model_inputs = batch.to(self.device_to_map_to)
                    print("Generating ...")
                    generated_ids = self.llm.model.generate(**model_inputs, max_new_tokens=512, pad_token_id=self.tokenizer.pad_id)
                    print("Decoding ...")
                    decoded_output = self.tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    print("Decoded.")
                    best_hypotheses.extend(decoded_output)
        return best_hypotheses

    def _build_prompts_dataset(self, dataset):
        prompts = self._build_prompts(dataset)
        return PromptsDataset(prompts)

    def _build_prompts(self, dataset) -> list[str]:
        prompts = []
        for sample_idx in range(dataset.get_num_of_samples()):
            prompt = self._generate_prompt(dataset, sample_idx)
            prompts.append(prompt)
        return prompts

    def _generate_prompt(self, dataset, sample_idx):
        beam_size = dataset.get_beam_size()

        hypotheses_list = ""
        from_idx = sample_idx * beam_size
        to_idx = (sample_idx + 1) * beam_size
        indexes = list(range(from_idx, to_idx))

        for i in range(beam_size):
            hypothesis_index = indexes[i]
            hypothesis = dataset[hypothesis_index][0]
            hypothesis_number = i + 1
            hypotheses_list += f'<hipoteza{hypothesis_number}> {hypothesis} </hipoteza{hypothesis_number}>\n'

        prompt_start = (
            f"Izvedi popravljanje napak na najboljših {beam_size} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (Automatic Speech Recognition). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost sistema ASR, so naslednje:\n\n"
        )
        prompt_end = "\n\nProsim, izpiši popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        return prompt_start + hypotheses_list + prompt_end

    def _sanitize_llm_output(self, output):
        return output.translate(str.maketrans('', '', string.punctuation)).lower()
