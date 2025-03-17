import argparse
import datetime
import json
import random
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import whisper
from tqdm import tqdm

from logger import Logger
from metrics_calculator import MetricsCalculator
from whisper.normalizers import BasicTextNormalizer

class WhisperTranscriber:
    def __init__(self, device_idx: int):
        self.device = f"cuda:{device_idx}"
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
    parser.add_argument('--save_frequency', type=int, default=100, help='After how many processed samples to save the results')
    parser.add_argument('--offset', type=int, default=0, help='Offset to skip samples.')
    args = parser.parse_args()

    with open(args.manifest_file_path, "r", encoding="utf-8") as f:
        read_lines = f.readlines()[args.offset:]

    split_lines = [read_lines[i:i + args.save_frequency] for i in range(0, len(read_lines), args.save_frequency)]

    def process_batch(args_tuple: tuple) -> tuple:
        manifest_lines, transcriber = args_tuple
        calc = MetricsCalculator()

        transcr_manifest_lines = []
        ignor_manifest_lines = []
        hyp_list = []
        scores_list = []
        wer_sum = 0

        for line in manifest_lines:
            manifest_entry = json.loads(line)
            try:
                best_texts, best_log_probs = transcriber.transcribe(manifest_entry["audio_filepath"], args.beam_width)

                transcr_manifest_lines.append(line)
                hyp_list.extend(best_texts)
                scores_list.extend(best_log_probs)

                current_wer = calc.calculate_wer([best_texts[0]], [manifest_entry["text"]])
                wer_sum += current_wer

                if args.log_results:
                    Logger.info(f" PATH: {manifest_entry['audio_filepath']}")
                    Logger.info(f"TRUTH: {manifest_entry['text']}")
                    Logger.info(f" BEST: {best_texts[0]}")
                    Logger.info(f"  WER: {current_wer}")
            except Exception as e:
                Logger.error(e)
                ignor_manifest_lines.append(line)

        return transcr_manifest_lines, ignor_manifest_lines, hyp_list, scores_list, wer_sum

    num_gpus = torch.cuda.device_count()
    device_indices = list(range(num_gpus))

    beams_file_path = f"{args.results_dir_path}/beams_{args.beam_width}.tsv"
    transcribed_manifest_path = f"{args.results_dir_path}/transcribed_manifest.nemo"
    ignored_manifest_path = f"{args.results_dir_path}/ignored_manifest.nemo"

    transcribed_manifest_lines, ignored_manifest_lines, hypotheses_list, asr_scores_list = [], [], [], []
    wer = 0

    start_time = time.time()
    Logger.info(f"Starting processing...")
    with Pool(num_gpus) as p:
        transcribers = [WhisperTranscriber(device_idx) for device_idx in device_indices]
        for idx, lines in enumerate(tqdm(split_lines)):
            Logger.info(f"Processing batch {idx + 1}/{len(split_lines)}...")

            batched_lines = [list(batch) for batch in np.array_split(lines, num_gpus)]

            Logger.info(f"Running decoding in {num_gpus} processes...")
            results = p.map(process_batch, list(zip(batched_lines, transcribers)))

            Logger.info(f"Saving results...")
            for result in results:
                transcribed_manifest_lines.extend(result[0])
                ignored_manifest_lines.extend(result[1])
                hypotheses_list.extend(result[2])
                asr_scores_list.extend(result[3])
                wer += result[4]

            with open(transcribed_manifest_path, "w", encoding="utf-8") as transcribed_manifest_file:
                transcribed_manifest_file.writelines(transcribed_manifest_lines)
            with open(ignored_manifest_path, "w", encoding="utf-8") as ignored_manifest_file:
                ignored_manifest_file.writelines(ignored_manifest_lines)

            df = pd.DataFrame({"hypotheses": hypotheses_list, "asr_scores": asr_scores_list})
            df.to_csv(beams_file_path, sep='\t', index=False, header=False)

            Logger.info(f"Saved beams and manifests to {args.results_dir_path}")
            Logger.info(f'WER = {wer / len(transcribed_manifest_lines)}')

    run_duration = time.time() - start_time
    Logger.info(f"Completed in {str(datetime.timedelta(seconds=run_duration))}")

