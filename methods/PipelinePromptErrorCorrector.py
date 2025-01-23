import string

from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from Logger import Logger
from methods.Method import Method
from torch_datasets.PromptsDataset import PromptsDataset
from torch_datasets.HypothesesDataset import HypothesesDataset
from utils.are_chat_templates_supported import are_chat_templates_supported


class PipelinePromptErrorCorrector(Method):
    def __init__(self, llm_name: str, tokenizer_name: str, batch_size: int = 8):
        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llm_name,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding_side="left")
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._generator = pipeline(
            "text-generation",
            model=llm,
            tokenizer=self._tokenizer,
            device_map="auto",
            torch_dtype="auto",
            num_return_sequences=1,
            max_new_tokens=256,
            return_full_text=False,
        )
        self._batch_size = batch_size
        self._should_use_chat_templates = are_chat_templates_supported(self._tokenizer)

    def run(self, dataset: HypothesesDataset) -> list[str]:
        prompts_dataset = self._build_prompts_dataset(dataset)
        Logger.info(f"{len(prompts_dataset)} prompts built.")
        Logger.info(f"Example prompt: {prompts_dataset[0]}")
        Logger.info("Correcting hypotheses ...")
        best_hypotheses = []
        for sequences in tqdm(
                self._generator(
                    prompts_dataset,
                    padding=True,  # Pad to the longest sequence in the batch
                    batch_size=self._batch_size
                ),
                total=len(prompts_dataset)
        ):
            generated_text = sequences[-1]["generated_text"]
            sanitized_text = self._sanitize_llm_output(generated_text)
            best_hypotheses.append(sanitized_text)
        return best_hypotheses

    def _build_prompts_dataset(self, dataset):
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
        return PromptsDataset(prompts)

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
