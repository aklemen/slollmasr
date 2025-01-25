import gc
import string

import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from Logger import Logger
from torch_datasets.HypothesesDataset import HypothesesDataset
from torch_datasets.PromptsDataset import PromptsDataset
from utils.are_chat_templates_supported import are_chat_templates_supported


class PromptErrorCorrector:
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 8):
        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llm_name,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding_side="left")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generator = pipeline(
            "text-generation",
            model=llm,
            tokenizer=self._tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            num_return_sequences=1,
            max_new_tokens=256,
            return_full_text=False,
        )
        self._batch_size = batch_size
        self._should_use_chat_templates = are_chat_templates_supported(self._tokenizer)

    def run(self, dataset: HypothesesDataset) -> list[str]:
        original_indices, sorted_prompts = self._build_prompts_list(dataset)
        last_best_hypotheses = [""] * len(sorted_prompts)
        last_processed_idx = 0
        return self._generate_hypotheses(
            sorted_prompts,
            original_indices,
            self._batch_size,
            last_best_hypotheses,
            last_processed_idx,
        )

    def _generate_hypotheses(
            self,
            sorted_prompts: list[str],
            original_indices: list[int],
            batch_size: int,
            last_best_hypotheses: list[str],
            last_processed_idx: int
    ) -> list[str]:
        Logger.info("Generating corrected hypotheses ...")
        unprocessed_sorted_dataset = PromptsDataset(sorted_prompts[last_processed_idx:])
        unprocessed_original_indices = original_indices[last_processed_idx:]
        try:
            for idx, output in enumerate(tqdm(
                self._generator(unprocessed_sorted_dataset, padding=True, batch_size=batch_size),
                total=len(unprocessed_sorted_dataset)
            )):
                generated_text = output[-1]["generated_text"]
                sanitized_text = self._sanitize_llm_output(generated_text)
                original_index = unprocessed_original_indices[idx]
                last_best_hypotheses[original_index] = sanitized_text
                last_processed_idx += 1
        except torch.cuda.OutOfMemoryError as e:
            Logger.warn("Ran out of GPU memory! Freeing GPU memory ...")
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                Logger.warn("Batch size is already 1, cannot reduce it further.")
                raise e
            if batch_size == self._batch_size / 2:
                Logger.info(f"Batch size was already halved once.")
                new_batch_size = 1
            else:
                new_batch_size = batch_size // 2
            Logger.info(f"Trying again with half the batch size {new_batch_size} ...")
            return self._generate_hypotheses(sorted_prompts, original_indices, new_batch_size, last_best_hypotheses, last_processed_idx)
        return last_best_hypotheses

    def _build_prompts_list(self, dataset):
        prompts = self._build_prompts(dataset)
        Logger.info(f"{len(prompts)} prompts built.")
        Logger.info(f"First prompt: {prompts[0]}")
        Logger.info(f"First response: {self._generator(prompts[0])[-1]['generated_text']}")
        Logger.info("Sorting prompts ...")

        sorted_prompts_with_indices = sorted(enumerate(prompts), key=lambda x: len(x[1]))
        original_indices = [prompt[0] for prompt in sorted_prompts_with_indices]
        sorted_prompts = [prompt[1] for prompt in sorted_prompts_with_indices]

        Logger.info(f"First sorted prompt: {sorted_prompts[0]}")
        Logger.info(f"First sorted response: {self._generator(sorted_prompts[0])[-1]['generated_text']}")
        Logger.info("Creating prompts dataset ...")

        return original_indices, sorted_prompts

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