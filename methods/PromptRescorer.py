from transformers import pipeline, AutoTokenizer

from methods.Method import Method
from torch_datasets.HypothesesDataset import HypothesesDataset
from utils.are_chat_templates_supported import are_chat_templates_supported


# !!! - WIP
class PromptRescorer(Method):
    def __init__(self, llm_name: str, tokenizer_name: str):
        self._generator = pipeline(
            "text-generation",
            model=llm_name,
            tokenizer=tokenizer_name,
            device_map="auto",
            torch_dtype="auto",
            num_return_sequences=1,
            max_new_tokens=512,
            return_full_text=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self._should_use_chat_templates = are_chat_templates_supported(tokenizer)
        self._base_prompt = (
            "Prejel boš seznam hipotez in njihovih ocen iz sistema ASR (Automatic Speech Recognition). "
            "Ponovno oceni hipoteze in izpiši le nove ocene, ločene z vejico, na primer '1.23,3.24,0.12,4.56,2.34'. "
            "Seznam hipotez in njihovih ocen je naslednji:\n\n"
        )

    def run(self, dataset: HypothesesDataset) -> list[float]:
        beam_size = dataset.get_beam_size()

        new_scores = []
        for sample_idx in range(dataset.get_num_of_samples()):
            prompt = self._generate_prompt(dataset, sample_idx)
            sequences = self._generator(prompt)
            result = sequences[-1]["generated_text"]
            partial_new_scores = ','.split(result)
            if len(partial_new_scores) != beam_size:
                raise Exception(
                    f"Expected {beam_size} scores, but got list of length {len(partial_new_scores)}. Result from LLM: {result}")
            partial_new_scores = [float(score) for score in partial_new_scores]
            new_scores.append(partial_new_scores)

        return [item for sublist in new_scores for item in sublist]

    def _generate_prompt(self, dataset, sample_idx):
        hypotheses_list = ""
        from_idx = sample_idx * dataset.get_beam_size()
        to_idx = (sample_idx + 1) * dataset.get_beam_size()
        for i in range(from_idx, to_idx):
            hypotheses_list += f'<hipoteza>{dataset[i][0]}</hipoteza><ocena>{str(dataset[i][1])}</ocena>\n'

        prompt = self._base_prompt + hypotheses_list

        if self._should_use_chat_templates:
            return [{ "role": "user", "content": prompt }]
        return prompt