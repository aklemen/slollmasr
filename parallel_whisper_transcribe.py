import datetime
import time

import torch
import whisper
from whisper.normalizers import BasicTextNormalizer

# Global model variable for multiprocessing
_model = None

def load_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper.load_model(name="base", device=device)
    return _model

class WhisperTranscriber:
    def __init__(self):
        self.model = load_model()
        self.normalizer = BasicTextNormalizer()

    def transcribe(self, audio_filepath: str, beam_width: int):
        audio = whisper.load_audio(audio_filepath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        options = whisper.DecodingOptions(language="sl", beam_size=50)

        texts, log_probs = whisper.decode(self.model, mel, options)
        texts = [self.normalizer(text) for text in texts]

        return texts, log_probs


import argparse
import json
import os
import pandas as pd
import random
import torch
import whisper
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from logger import Logger
from metrics_calculator import MetricsCalculator

# Number of parallel processes (adjust based on your GPU capacity)
NUM_WORKERS = 20


def process_manifest_entry(entry):
    """ Process a single manifest entry for transcription """
    transcriber = WhisperTranscriber()
    try:
        best_texts, best_log_probs = transcriber.transcribe(entry["audio_filepath"], entry["beam_width"])
        current_wer = MetricsCalculator().calculate_wer([best_texts[0]], [entry["text"]])
        return {
            "audio_filepath": entry["audio_filepath"],
            "truth": entry["text"],
            "best_texts": best_texts,
            "best_log_probs": best_log_probs,
            "wer": current_wer,
            "success": True
        }
    except Exception as e:
        return {"audio_filepath": entry["audio_filepath"], "error": str(e), "success": False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create hypotheses-to-transcript mapping dataset with Whisper')
    parser.add_argument('--manifest_file_path', type=str, help='Path to the NeMo manifest file')
    parser.add_argument('--results_dir_path', type=str, help='Path to output the resulting beams and manifests')
    parser.add_argument('--beam_width', type=int, help='Width of the resulting beams')
    parser.add_argument('--log_results', type=bool, default=False,
                        help='Logs the path, best hypothesis, ground truth and WER')
    args = parser.parse_args()

    with open(args.manifest_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Prepare manifest data for parallel processing
    manifest_entries = []
    for line in lines:
        entry = json.loads(line)
        entry["beam_width"] = args.beam_width  # Add beam width to entry for processing
        manifest_entries.append(entry)

    start_time = time.time()
    # Run parallel transcription
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_manifest_entry, manifest_entries), total=len(manifest_entries)))

    # Process results
    transcribed_manifest_lines = []
    ignored_manifest_lines = []
    hypotheses_list = []
    asr_scores_list = []
    total_wer = 0
    processed_count = 0

    for result in results:
        if result["success"]:
            transcribed_manifest_lines.append(json.dumps(result) + "\n")
            hypotheses_list.extend(result["best_texts"])
            asr_scores_list.extend(result["best_log_probs"])
            total_wer += result["wer"]
            processed_count += 1
            if args.log_results:
                Logger.info(f" PATH: {result['audio_filepath']}")
                Logger.info(f"TRUTH: {result['truth']}")
                Logger.info(f" BEST: {result['best_texts'][0]}")
                Logger.info(f"  WER: {result['wer']}")
        else:
            Logger.error(f"Error processing {result['audio_filepath']}: {result['error']}")
            ignored_manifest_lines.append(result["audio_filepath"] + "\n")

    # Save results
    os.makedirs(args.results_dir_path, exist_ok=True)

    beams_file_path = os.path.join(args.results_dir_path, f"beams_{args.beam_width}.tsv")
    df = pd.DataFrame({"hypotheses": hypotheses_list, "asr_scores": asr_scores_list})
    df.to_csv(beams_file_path, sep='\t', index=False, header=False)

    transcribed_manifest_path = os.path.join(args.results_dir_path, "transcribed_manifest.nemo")
    ignored_manifest_path = os.path.join(args.results_dir_path, "ignored_manifest.nemo")

    with open(transcribed_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(transcribed_manifest_lines)
    with open(ignored_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(ignored_manifest_lines)

    Logger.info(f"Saved beams and manifests to {args.results_dir_path}")
    Logger.info(f'WER = {total_wer / processed_count if processed_count > 0 else 0}')

    run_duration = time.time() - start_time
    Logger.info(f"Completed in {str(datetime.timedelta(seconds=run_duration))}")
