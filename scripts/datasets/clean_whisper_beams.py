import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer

from logger import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whisper_manifest_file_path', type=str, required=True)
    parser.add_argument('--whisper_beams_file_path', type=str, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)

    arguments = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(arguments)
    Logger.info("===================================")
    return arguments


def read_manifest(manifest_file_path):
    with open(manifest_file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def read_grouped_hypotheses(beams_file_path, beam_size):
    hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    hypotheses = hypotheses["text"].tolist()
    hypotheses = [str(text) for text in hypotheses]
    return [hypotheses[i:i + beam_size] for i in range(0, len(hypotheses), beam_size)]


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir_path, exist_ok=True)

    model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(text):
        return len(tokenizer.encode(text, truncation=False))

    whisper_manifest = read_manifest(args.whisper_manifest_file_path)
    whisper_hypotheses = read_grouped_hypotheses(args.whisper_beams_file_path, 10)

    utterance_with_max_transcript_tokens = max(whisper_manifest, key=lambda x: count_tokens(x["text"]))

    Logger.info(f"Utterance with max transcript tokens: {utterance_with_max_transcript_tokens}")

    safety_buffer = 50

    beam_size = 10
    max_tokens = count_tokens(utterance_with_max_transcript_tokens["text"]) * beam_size
    max_tokens += safety_buffer

    Logger.info(f"Max tokens to accept: {max_tokens} (safety buffer: {safety_buffer})")

    whisper_indices_to_remove = [
        i for i, hypotheses in enumerate(whisper_hypotheses)
        if count_tokens("".join(hypotheses)) > max_tokens
    ]

    Logger.info(f"Number of hypotheses to remove: {len(whisper_indices_to_remove)}")

    manifest_output_path = os.path.join(args.output_dir_path, "whisper_manifest_cleaned.nemo")
    with open(manifest_output_path, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(whisper_manifest):
            if i not in whisper_indices_to_remove:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    beams_indices_to_remove = []
    for index in whisper_indices_to_remove:
        beams_indices_to_remove.extend(range(index * beam_size, (index + 1) * beam_size))
    beams_output_path = os.path.join(args.output_dir_path, "whisper_beams_cleaned.tsv")
    df_hypotheses = pd.read_csv(args.whisper_beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    df_cleaned = df_hypotheses.drop(beams_indices_to_remove)
    df_cleaned.to_csv(beams_output_path, sep="\t", header=False, index=False)

    Logger.info(f"Cleaned files saved to: {manifest_output_path} and {beams_output_path}")