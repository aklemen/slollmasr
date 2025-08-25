import time

import lhotse.dataset
import torch
from lhotse import CutSet
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2 import SALM
from transformers import GenerationConfig

from prompting.sanitize_string import sanitize_string


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}

class SalmSpeechLM:
    def __init__(self, llm_name_or_path: str, batch_size: int = 8):
        self.model = SALM.from_pretrained(llm_name_or_path).eval().to(torch.bfloat16).to("cuda")
        self.batch_size = batch_size

    def run(self, manifest_file_path: str) -> tuple[list[str], float]:
        cuts = guess_parse_cutset(manifest_file_path)
        for idx, cut in enumerate(cuts):
            cut.custom["original_index"] = idx
        cuts = cuts.sort_by_duration()

        dloader = torch.utils.data.DataLoader(
            dataset=ToAudio(),
            sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=self.batch_size),
            num_workers=1,
            batch_size=None,
        )

        user_prompt = "Transkribiraj naslednji posnetek:"
        prompt = [{"role": "user", "content": f"{user_prompt} {self.model.audio_locator_tag}"}]

        start_time = time.time()
        hypotheses = []
        for batch_idx, batch in enumerate(dloader):
            batch_answer_ids = self.model.generate(
                prompts=[prompt] * len(batch["cuts"]),  # identical prompt for each example
                audios=batch["audios"].to(self.model.device, non_blocking=True),
                audio_lens=batch["audio_lens"].to(self.model.device, non_blocking=True),
                generation_config=GenerationConfig(
                    max_new_tokens=256,
                    bos_token_id=self.model.text_bos_id,
                    eos_token_id=self.model.text_eos_id,
                    pad_token_id=self.model.text_pad_id,
                ),
            )
            batch_answer_ids = batch_answer_ids.cpu()
            batch_hypotheses = []
            for answer_ids in batch_answer_ids:
                answer_text = self.model.tokenizer.ids_to_text(self._truncate_to_eos(answer_ids))
                sanitized_answer_text = sanitize_string(answer_text)
                batch_hypotheses.append(sanitized_answer_text)

            hypotheses.extend(batch_hypotheses)
        inference_time = time.time() - start_time

        original_indices = [cut.custom["original_index"] for cut in cuts]
        ordered_hypotheses = ["EMPTY HYPOTHESIS"] * len(hypotheses)
        for idx, orig_idx in enumerate(original_indices):
            ordered_hypotheses[orig_idx] = hypotheses[idx]

        return ordered_hypotheses, inference_time


    def _truncate_to_eos(self, answer: torch.Tensor):
        end = torch.isin(answer, torch.tensor([self.model.text_eos_id])).nonzero(as_tuple=True)[0]
        if end.numel() == 0:
            return answer
        end = end[0]
        return answer[:end]