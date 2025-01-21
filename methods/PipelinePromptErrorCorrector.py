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

    def run(self, dataset: HypothesesDataset) -> list[str]:
        prompts_dataset = self._build_prompts_dataset(dataset)
        Logger.info(f"{len(prompts_dataset)} prompts built. Generating ...")
        best_hypotheses = []
        for sequences in tqdm(
                self._generator(
                    prompts_dataset,
                    padding=True,
                    batch_size=self._batch_size
                ),
                total=len(prompts_dataset)
        ):
            output = sequences[-1]["generated_text"]
            sanitized_result = self._sanitize_llm_output(output)
            best_hypotheses.append(sanitized_result)
        return best_hypotheses

    def _build_prompts_dataset(self, dataset):
        prompts = []
        for sample_idx in range(dataset.get_num_of_samples()):
            prompt = self._generate_prompt(dataset, sample_idx)
            prompts.append(prompt)
        if are_chat_templates_supported(self._tokenizer):
            prompts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in prompts
            ]
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
        prompt_end = "\n\nProsim, izpiši le popravljen najboljši transkript danega govora, brez dodatnih razlag ali besed."

        return prompt_start + hypotheses_list + prompt_end

    def _sanitize_llm_output(self, output):
        return output.translate(str.maketrans('', '', string.punctuation)).lower()
