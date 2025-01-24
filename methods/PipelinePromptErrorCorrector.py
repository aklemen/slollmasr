import gc
import string

import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from Logger import Logger
from methods.Method import Method
from torch_datasets.PromptsDataset import PromptsDataset
from torch_datasets.HypothesesDataset import HypothesesDataset
from utils.are_chat_templates_supported import are_chat_templates_supported


class PipelinePromptErrorCorrector(Method):
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 8):
        self._llm_name = llm_name
        self._tokenizer_name = tokenizer_name
        self._generator, self._tokenizer = self._create_generator_and_tokenizer(llm_name, tokenizer_name)
        self._batch_size = batch_size
        self._should_use_chat_templates = are_chat_templates_supported(self._tokenizer)

    def run(self, dataset: HypothesesDataset) -> list[str]:
        original_indices, sorted_dataset = self._build_prompts_dataset(dataset)
        return self._generate_hypotheses(sorted_dataset, original_indices, self._batch_size)

    def _generate_hypotheses(self, sorted_dataset: PromptsDataset, original_indices: list[int], batch_size: int) -> list[str]:
        Logger.info("Generating corrected hypotheses ...")
        best_hypotheses = [""] * len(sorted_dataset)
        try:
            for idx, output in enumerate(tqdm(
                self._generator(sorted_dataset, padding=True, batch_size=batch_size),
                total=len(sorted_dataset)
            )):
                generated_text = output[-1]["generated_text"]
                sanitized_text = self._sanitize_llm_output(generated_text)
                original_index = original_indices[idx]
                best_hypotheses[original_index] = sanitized_text
        except torch.cuda.OutOfMemoryError as e:
            Logger.warn("Ran out of GPU memory!")
            self._release_gpu_memory()
            new_batch_size = batch_size // 2
            if new_batch_size == 0:
                Logger.warn("Cannot retry as batch size is already 0.")
                raise e
            Logger.info(f"Trying again with half the batch size ({new_batch_size}) ...")
            return self._generate_hypotheses(sorted_dataset, original_indices, new_batch_size)
        return best_hypotheses

    def _build_prompts_dataset(self, dataset):
        prompts = self._build_prompts(dataset)
        Logger.info(f"{len(prompts)} prompts built.")
        Logger.info(f"First prompt: {prompts[0]}")
        Logger.info(f"First response: {self._generator(prompts[0])[-1]['generated_text']}")
        Logger.info("Sorting prompts ...")

        sorted_prompts_with_indices = sorted(enumerate(prompts), key=lambda x: len(x[1]), reverse=True)
        original_indices = [prompt[0] for prompt in sorted_prompts_with_indices]
        sorted_prompts = [prompt[1] for prompt in sorted_prompts_with_indices]

        Logger.info(f"First sorted prompt: {sorted_prompts[0]}")
        Logger.info(f"First sorted response: {self._generator(sorted_prompts[0])[-1]['generated_text']}")
        Logger.info("Creating prompts dataset ...")

        sorted_prompts_dataset = PromptsDataset(sorted_prompts)

        return original_indices, sorted_prompts_dataset

    def _build_prompts(self, dataset):
        pre_prompt = (
            f"Izvedi popravljanje napak na najboljših {dataset.get_beam_size()} izhodih, ki jih je generiral sistem za samodejno razpoznavanje govora (Automatic Speech Recognition). "
            f"Hipoteze, navedene po vrstnem redu glede na njihovo posteriorno verjetnost sistema ASR, so naslednje:\n\n"
        )
        post_prompt = "\n\nProsim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        prompts = []
        for sample_idx in range(dataset.get_num_of_samples()):
            hypotheses_list = self._generate_hypotheses_list_for_prompt(dataset, sample_idx)
            prompt = self._transform_prompt_to_chat_if_supported(pre_prompt + hypotheses_list + post_prompt)
            prompts.append(prompt)
        return prompts

    def _generate_hypotheses_list_for_prompt(self, dataset, sample_idx):
        beam_size = dataset.get_beam_size()
        from_dataset_idx = sample_idx * beam_size
        to_dataset_idx = (sample_idx + 1) * beam_size
        hypotheses_list = [
            f'<hipoteza{i + 1}> {dataset[dataset_idx][0]} </hipoteza{i + 1}>'
            for i, dataset_idx in enumerate(range(from_dataset_idx, to_dataset_idx))
        ]
        return "\n".join(hypotheses_list)

    def _transform_prompt_to_chat_if_supported(self, prompt):
        if self._should_use_chat_templates:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt

    def _sanitize_llm_output(self, text: str) -> str:
        return text.strip().translate(str.maketrans('', '', string.punctuation)).lower()

    def _create_generator_and_tokenizer(self, llm_name: str, tokenizer_name: str):
        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llm_name,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        generator = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype="auto",
            num_return_sequences=1,
            max_new_tokens=256,
            return_full_text=False,
        )
        return generator, tokenizer

    def _release_gpu_memory(self):
        Logger.info("Freeing GPU memory ...")
        self._generator, self._tokenizer = self._create_generator_and_tokenizer(self._llm_name, self._tokenizer_name)
        gc.collect()
        torch.cuda.empty_cache()
