import argparse
import json
import os
import re
from collections import Counter

import pandas as pd
from tqdm import tqdm
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

def detect_repetition(text, min_repeat_count=5, max_char_repeat=5, min_pattern_repeats=10):
    words = text.lower().split()
    word_counts = Counter(words)
    repeated_words = {word: count for word, count in word_counts.items() if count >= min_repeat_count}

    consecutive_repeats = []
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            consecutive_repeats.append(words[i])

    char_repeats = re.findall(r'(.)\1{' + str(max_char_repeat) + ',}', text.lower())

    pattern_repeats = []
    for pattern_len in range(2, 6):
        # Find all starting positions of patterns
        for start in range(len(text) - pattern_len * min_pattern_repeats + 1):
            pattern = text[start:start + pattern_len].lower()

            # Count consecutive repetitions starting at this position
            count = 0
            pos = start
            while pos + pattern_len <= len(text) and text[pos:pos + pattern_len].lower() == pattern:
                count += 1
                pos += pattern_len

            # If we found enough repetitions, add to results
            if count >= min_pattern_repeats:
                pattern_repeats.append(pattern)
                break  # Move to next pattern length to avoid duplicates

    unique_words = len(set(words))
    total_words = len(words)
    repetition_ratio = 1 - (unique_words / total_words) if total_words > 0 else 0

    return {
        'has_repetition': bool(repeated_words or consecutive_repeats or char_repeats or pattern_repeats),
        'repeated_words': repeated_words,
        'consecutive_repeats': list(set(consecutive_repeats)),
        'char_repeats': char_repeats,
        'pattern_repeats': pattern_repeats,
        'repetition_ratio': repetition_ratio,
        'is_highly_repetitive': repetition_ratio > 0.7
    }


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

    whisper_indices_to_remove = []
    for i, hypotheses in tqdm(enumerate(whisper_hypotheses)):
        total_tokens = count_tokens("\n".join(hypotheses))
        repetition_info = detect_repetition(hypotheses[0])

        if total_tokens > max_tokens or repetition_info['has_repetition'] or repetition_info['is_highly_repetitive']:
            whisper_indices_to_remove.append(i)

    Logger.info(f"Number of hypotheses to remove: {len(whisper_indices_to_remove)}")

    manifest_output_path = os.path.join(args.output_dir_path, "whisper_manifest_cleaned.nemo")
    removes_manifest_output_path = os.path.join(args.output_dir_path, "whisper_manifest_removed.nemo")

    with open(manifest_output_path, 'w', encoding='utf-8') as clean_f, \
            open(removes_manifest_output_path, 'w', encoding='utf-8') as removed_f:
        for i, entry in enumerate(whisper_manifest):
            json_line = json.dumps(entry, ensure_ascii=False) + '\n'
            if i in whisper_indices_to_remove:
                removed_f.write(json_line)
            else:
                clean_f.write(json_line)

    beams_indices_to_remove = []
    for index in whisper_indices_to_remove:
        beams_indices_to_remove.extend(range(index * beam_size, (index + 1) * beam_size))

    beams_output_path = os.path.join(args.output_dir_path, "whisper_beams_cleaned.tsv")
    removed_beams_output_path = os.path.join(args.output_dir_path, "whisper_beams_removed.tsv")

    df_hypotheses = pd.read_csv(args.whisper_beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    df_cleaned = df_hypotheses.drop(beams_indices_to_remove)
    df_cleaned.to_csv(beams_output_path, sep="\t", header=False, index=False)
    df_removed = df_hypotheses.iloc[beams_indices_to_remove]
    df_removed.to_csv(removed_beams_output_path, sep="\t", header=False, index=False)

    Logger.info(f"Cleaned files saved to: {manifest_output_path} and {beams_output_path}")