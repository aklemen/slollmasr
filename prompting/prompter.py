import gc
from typing import Union

import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from utils.logger import Logger
from prompting.prompts_dataset import PromptsDataset
from prompting.sanitize_string import sanitize_string


class Prompter:
    def __init__(self, llm_name_or_path: str, tokenizer_name_or_path: str, batch_size: int = 8):
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            Logger.info(f"No pad_token available. Setting pad_token to eos_token: {self._tokenizer.eos_token}")
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llm_name_or_path,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            num_return_sequences=1,
            max_new_tokens=256,
            return_full_text=False,
        )

        self._batch_size = batch_size

    def execute_prompts(self, prompts_or_chats: list[Union[str, list[dict[str, str]]]]) -> list[str]:
        is_chat = isinstance(prompts_or_chats[0], list)
        if is_chat:
            Logger.info("Detected chat format, applying chat template to transform chats to prompts ...")
            if not self._are_chat_templates_supported(self._tokenizer):
                raise Exception(
                    f"Chat templates are not supported by the given tokenizer: {self._tokenizer.name_or_path}.")
            prompts = [self._transform_chat_to_prompt(chat) for chat in prompts_or_chats]
        else:
            Logger.info(f"Prompts are not in chat format, so they will be kept as is. First sample: {prompts_or_chats[0]}")
            prompts = prompts_or_chats
        original_indices, sorted_prompts = self._sort_prompts(prompts)
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
        Logger.info("Generating new best hypotheses ...")
        unprocessed_sorted_dataset = PromptsDataset(sorted_prompts[last_processed_idx:])
        unprocessed_original_indices = original_indices[last_processed_idx:]
        try:
            for idx, output in enumerate(tqdm(
                    self._generator(unprocessed_sorted_dataset, padding=True, batch_size=batch_size),
                    total=len(unprocessed_sorted_dataset)
            )):
                generated_text = output[-1]["generated_text"]
                sanitized_text = sanitize_string(generated_text)
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
            Logger.info(f"Trying again with new batch size {new_batch_size} ...")
            return self._generate_hypotheses(
                sorted_prompts,
                original_indices,
                new_batch_size,
                last_best_hypotheses,
                last_processed_idx
            )
        return last_best_hypotheses

    def _transform_chat_to_prompt(self, chat: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _sort_prompts(self, prompts: list[str]) -> tuple[list[int], list[str]]:
        Logger.info(f"First prompt: {prompts[0]}")
        Logger.info(f"First response: {self._generator(prompts[0])[-1]['generated_text']}")

        Logger.info("Sorting prompts ...")
        sorted_prompts_with_indices = sorted(enumerate(prompts), key=lambda x: len(x[1]))
        original_indices = [prompt[0] for prompt in sorted_prompts_with_indices]
        sorted_prompts = [prompt[1] for prompt in sorted_prompts_with_indices]

        Logger.info(f"First sorted prompt: {sorted_prompts[0]}")
        Logger.info(f"First sorted response: {self._generator(sorted_prompts[0])[-1]['generated_text']}")

        return original_indices, sorted_prompts

    def _are_chat_templates_supported(self) -> bool:
        return hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template is not None