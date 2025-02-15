import argparse
import json
import random

import pandas as pd
import torch
import whisper
from tqdm import tqdm

from metrics_calculator import MetricsCalculator

def get_top_n_indices(top_hypotheses: list[str], n: int) -> list[int]:
    top_n_indices: list[int] = []

    top_n_hypotheses: list[str] = []
    for idx, hypothesis in enumerate(top_hypotheses):
        if len(top_n_indices) < n and len(hypothesis) > 0 and hypothesis not in top_n_hypotheses:
            top_n_hypotheses.append(hypothesis)
            top_n_indices.append(idx)

    if len(top_n_indices) < n:
        for _ in range(n - len(top_n_hypotheses)):
            random_idx_to_use_again = random.choice(top_n_indices)
            top_n_indices.append(random_idx_to_use_again)

    return top_n_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create hypotheses-to-transcript mapping dataset with Whisper')
    parser.add_argument('--manifest_file_path', type=str, help='Path to the NeMo manifest file')
    parser.add_argument('--beams_file_path', type=str, help='Path to output the resulting beams')
    parser.add_argument('--beam_width', type=int, help='Width of the resulting beams')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(name="base", device=device)
    # normalizer = BasicTextNormalizer()
    calc = MetricsCalculator()

    hypotheses_list = []
    asr_scores_list = []
    count = 0
    wer = 0

    with open(args.manifest_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        manifest_entry = json.loads(line)
        audio = whisper.load_audio(manifest_entry["audio_filepath"])
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language="sl", beam_size=50)
        texts, log_probs = whisper.decode(model, mel, options)

        print(f"=" * 10, "Texts", "=" * 10)
        print(texts)
        print(f"=" * 10, "Log probs", "=" * 10)
        print(texts)

        #TODO - order by log_probs

        indices = get_top_n_indices(texts, args.beam_width)

        hypotheses_list.extend([texts[i] for i in indices])
        asr_scores_list.extend([log_probs[i] for i in indices])

        print(f"=" * 10, "Top Texts", "=" * 10)
        print(texts)
        print(f"=" * 10, "Top Log probs", "=" * 10)
        print(texts)

        count += 1
        current_wer = calc.calculate_wer([hypotheses_list[0]], [manifest_entry["text"]])
        wer += current_wer

    df = pd.DataFrame({"hypotheses": hypotheses_list, "asr_scores": asr_scores_list})
    df.to_csv(args.beams_file_path, sep='\t', index=False, header=False)

    print(f"Saved beams to {args.beams_file_path}")
    print(f'WER = {wer / count}')

