import string

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


class PromptErrorCorrector(Method):
    def __init__(self, llm: LargeLanguageModel, tokenizer: Tokenizer):
        super().__init__(llm, tokenizer)
        self._generator = pipeline(
            "text-generation",
            model=llm.model,
            tokenizer=tokenizer.tokenizer,
            device_map="auto",
            torch_dtype="auto",
            num_return_sequences=1,
            max_new_tokens=512,
            return_full_text=False,
        )
        self.should_use_chat_templates = tokenizer.are_chat_templates_supported()

    def run(self, dataset: HypothesesDataset) -> list[str]:
        prompts_dataset = self._build_prompts_dataset(dataset)
        best_hypotheses = []
        for sequences in tqdm(self._generator(prompts_dataset, batch_size=8)):
            output = sequences[-1]["generated_text"]
            sanitized_result = self._sanitize_llm_output(output)
            best_hypotheses.append(sanitized_result)
        return best_hypotheses


    def _build_prompts_dataset(self, dataset) -> PromptsDataset:
        prompts = []
        for sample_idx in range(dataset.get_num_of_samples()):
            prompt = self._generate_prompt(dataset, sample_idx)
            prompts.append(prompt)
        return PromptsDataset(prompts)

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

        prompt_end = "Prosim, izpiši popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        prompt = prompt_start + hypotheses_list + prompt_end

        if self.should_use_chat_templates:
            return [{ "role": "user", "content": prompt }]
        return prompt

    def _sanitize_llm_output(self, output):
        return output.translate(str.maketrans('', '', string.punctuation)).lower()
