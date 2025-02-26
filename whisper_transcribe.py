import argparse
import json
import random

import pandas as pd
import torch
import whisper
from tqdm import tqdm

from logger import Logger
from metrics_calculator import MetricsCalculator
from whisper.normalizers import BasicTextNormalizer

class WhisperTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(name="base", device=self.device)
        self.normalizer = BasicTextNormalizer()

    def transcribe(self, audio_filepath: str, beam_width: int) -> tuple[list[str], list[float]]:
        audio = whisper.load_audio(audio_filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        options = whisper.DecodingOptions(language="sl", beam_size=50)
        texts, log_probs = whisper.decode(self.model, mel, options)
        texts = [self.normalizer(text) for text in texts]

        indices = self._get_best_n_indices(texts, beam_width)

        unsorted_best_texts = [texts[i] for i in indices]
        unsorted_best_log_probs = [log_probs[i] for i in indices]

        sorted_pairs = sorted(zip(unsorted_best_texts, unsorted_best_log_probs), key=lambda x: x[1], reverse=True)
        sorted_best_texts, sorted_best_log_probs = [list(t) for t in zip(*sorted_pairs)]

        return sorted_best_texts, sorted_best_log_probs

    def _get_best_n_indices(self, best_hypotheses: list[str], n: int) -> list[int]:
        best_n_indices: list[int] = []

        best_n_hypotheses: list[str] = []
        for idx, hypothesis in enumerate(best_hypotheses):
            if len(best_n_indices) < n and len(hypothesis) > 0 and hypothesis not in best_n_hypotheses:
                best_n_hypotheses.append(hypothesis)
                best_n_indices.append(idx)

        if len(best_n_indices) == 0:
            raise Exception("All hypotheses are empty!")

        if len(best_n_indices) < n:
            for _ in range(n - len(best_n_hypotheses)):
                random_idx_to_use_again = random.choice(best_n_indices)
                best_n_indices.append(random_idx_to_use_again)

        return best_n_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create hypotheses-to-transcript mapping dataset with Whisper')
    parser.add_argument('--manifest_file_path', type=str, help='Path to the NeMo manifest file')
    parser.add_argument('--results_dir_path', type=str, help='Path to output the resulting beams and manifests')
    parser.add_argument('--beam_width', type=int, help='Width of the resulting beams')
    parser.add_argument('--log_results', type=bool, default=False, help='Logs the path, best hypothesis, ground truth and WER')
    args = parser.parse_args()

    with open(args.manifest_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    transcriber = WhisperTranscriber()
    calc = MetricsCalculator()

    transcribed_manifest_lines = []
    ignored_manifest_lines = []
    hypotheses_list = []
    asr_scores_list = []
    wer = 0

    for line in tqdm(lines):
        manifest_entry = json.loads(line)
        try:
            best_texts, best_log_probs = transcriber.transcribe(manifest_entry["audio_filepath"], args.beam_width)

            transcribed_manifest_lines.append(line)
            hypotheses_list.extend(best_texts)
            asr_scores_list.extend(best_log_probs)

            current_wer = calc.calculate_wer([best_texts[0]], [manifest_entry["text"]])
            wer += current_wer

            if args.log_results:
                Logger.info(f" PATH: {manifest_entry['audio_filepath']}")
                Logger.info(f"TRUTH: {manifest_entry['text']}")
                Logger.info(f" BEST: {best_texts[0]}")
                Logger.info(f"  WER: {current_wer}")
        except Exception as e:
            Logger.error(e)
            ignored_manifest_lines.append(line)

    beams_file_path = f"{args.results_dir_path}/beams_{args.beam_width}.tsv"
    df = pd.DataFrame({"hypotheses": hypotheses_list, "asr_scores": asr_scores_list})
    df.to_csv(beams_file_path, sep='\t', index=False, header=False)

    transcribed_manifest_path = f"{args.results_dir_path}/transcribed_manifest.nemo"
    ignored_manifest_path = f"{args.results_dir_path}/ignored_manifest.nemo"
    with open(transcribed_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(transcribed_manifest_lines)
    with open(ignored_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(ignored_manifest_lines)

    Logger.info(f"Saved beams and manifests to {args.results_dir_path}")
    Logger.info(f'WER = {wer / len(lines)}')

