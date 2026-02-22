#!/usr/bin/env python3
"""
Calculate average transcription tokens per second for the artur/clean/train dataset.
This helps verify if the modality adapter's subsampling factor is appropriate.
"""
import json
import gzip
from transformers import AutoTokenizer
import numpy as np

# Load dataset from NeMo manifest
manifest_path = "/shared/workspace/lpt-llm/datasets/artur/v1.0/nemo/clean/train.nemo"

print(f"Loading manifest from: {manifest_path}")

# Read NeMo manifest (JSONL format, possibly gzipped)
samples = []
opener = gzip.open if manifest_path.endswith('.gz') else open
with opener(manifest_path, 'rt') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Load tokenizers
gams_tokenizer = AutoTokenizer.from_pretrained("cjvt/GaMS-9B")
slovenian_gpt_tokenizer = AutoTokenizer.from_pretrained("aklemen/SlovenianGPT")

gams_tps = []
slovenian_tps = []
durations = []

print("Processing samples...")
count = 0
for sample in samples:
    text = sample.get('text', '')
    duration = sample.get('duration', 0)

    if not text or duration <= 0:
        continue

    gams_tokens = len(gams_tokenizer.encode(text))
    slovenian_tokens = len(slovenian_gpt_tokenizer.encode(text))

    gams_tps.append(gams_tokens / duration)
    slovenian_tps.append(slovenian_tokens / duration)
    durations.append(duration)

    count += 1
    if count % 10000 == 0:
        print(f"  Processed {count} samples...")

print(f"Finished processing {count} samples")


def print_stats(name, tps_list):
    tps = np.array(tps_list)
    avg_tps = np.mean(tps)
    # token_equivalent_duration = 1 / tokens_per_second
    avg_ted = 1 / avg_tps

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")
    print(f"Tokens per second:")
    print(f"  Mean:   {avg_tps:.2f}")
    print(f"  Std:    {np.std(tps):.2f}")
    print(f"  Min:    {np.min(tps):.2f}")
    print(f"  Max:    {np.max(tps):.2f}")
    print(f"  Median: {np.median(tps):.2f}")
    print(f"  P5:     {np.percentile(tps, 5):.2f}")
    print(f"  P95:    {np.percentile(tps, 95):.2f}")
    print(f"\nDerived token_equivalent_duration (1/avg_tps): {avg_ted:.3f}s")
    print(f"  → This would give {1 / avg_ted:.2f} audio tokens/sec")


print_stats("GaMS-9B (cjvt/GaMS-9B)", gams_tps)
print_stats("SlovenianGPT (aklemen/SlovenianGPT)", slovenian_tps)

print(f"\n{'=' * 50}")
print(f"Dataset statistics")
print(f"{'=' * 50}")
print(f"Total samples: {len(durations)}")
print(f"Duration - Mean: {np.mean(durations):.2f}s, Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s")
print(f"Total audio: {np.sum(durations) / 3600:.2f} hours")