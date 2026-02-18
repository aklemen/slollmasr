#!/usr/bin/env python3
"""
Estimate token-based bucketing configuration for SpeechLM2.

Analyzes a Lhotse shar dataset to determine optimal bucket_duration_bins
and bucket_batch_size for efficient training.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate token bucketing config for SpeechLM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shar-path",
        type=str,
        required=True,
        help="Path to Lhotse shar dataset directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer name or path",
    )
    parser.add_argument(
        "--token-equivalent-duration",
        type=float,
        required=True,
        help="Seconds per audio token (get from verify_token_duration.py)",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=10,
        help="Number of buckets to create",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to analyze (-1 for all)",
    )
    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=2,
        help="Known working batch size at median token count",
    )
    parser.add_argument(
        "--base-memory-usage",
        type=float,
        default=0.6,
        help="GPU memory fraction used at base batch size (0-1)",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum batch size per bucket",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.85,
        help="Safety factor for batch size estimation (0-1)",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_name: str):
    """Load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def load_lhotse_shar(shar_path: str, num_samples: int):
    """Load cuts from Lhotse shar dataset."""
    from lhotse import CutSet

    cuts = CutSet.from_shar(in_dir=shar_path)

    if num_samples > 0:
        from itertools import islice

        cuts = list(islice(cuts, num_samples))
    else:
        cuts = list(cuts)

    return cuts


def calculate_token_counts(
    cuts: list,
    tokenizer,
    token_equivalent_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate audio, text, and total token counts for each cut."""
    audio_tokens_list = []
    text_tokens_list = []
    total_tokens_list = []

    for cut in tqdm(cuts, desc="Calculating token counts"):
        duration = cut.duration
        audio_tokens = duration / token_equivalent_duration

        text = ""
        if cut.supervisions:
            text = cut.supervisions[0].text or ""

        text_tokens = len(tokenizer.encode(text)) if text else 0

        total_tokens = audio_tokens + text_tokens

        audio_tokens_list.append(audio_tokens)
        text_tokens_list.append(text_tokens)
        total_tokens_list.append(total_tokens)

    return (
        np.array(audio_tokens_list),
        np.array(text_tokens_list),
        np.array(total_tokens_list),
    )


def estimate_bucket_bins(total_tokens: np.ndarray, num_buckets: int) -> list[int]:
    """Estimate bucket bins using percentiles for even distribution."""
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:]
    bins = np.percentile(total_tokens, percentiles)
    bins = [int(np.ceil(b)) for b in bins]
    bins = sorted(set(bins))
    return bins


def estimate_batch_sizes(
    bucket_bins: list[int],
    median_tokens: float,
    base_batch_size: int,
    base_memory_usage: float,
    min_batch_size: int,
    safety_factor: float,
) -> list[int]:
    """
    Estimate batch size for each bucket.

    Memory scales roughly linearly with total tokens in batch.
    At base_batch_size with samples near median tokens, we use base_memory_usage.
    """
    batch_sizes = []

    available_memory_factor = 1.0 / base_memory_usage

    for bin_max in bucket_bins:
        tokens_ratio = median_tokens / bin_max
        estimated_batch = base_batch_size * available_memory_factor * tokens_ratio * safety_factor
        batch_size = max(min_batch_size, int(estimated_batch))
        batch_sizes.append(batch_size)

    return batch_sizes


def print_statistics(
    audio_tokens: np.ndarray,
    text_tokens: np.ndarray,
    total_tokens: np.ndarray,
):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nSamples analyzed: {len(total_tokens)}")

    print(f"\nAudio tokens:")
    print(f"  Min: {audio_tokens.min():.1f}, Max: {audio_tokens.max():.1f}")
    print(f"  Mean: {audio_tokens.mean():.1f}, Std: {audio_tokens.std():.1f}")
    print(f"  Median: {np.median(audio_tokens):.1f}")

    print(f"\nText tokens:")
    print(f"  Min: {text_tokens.min():.0f}, Max: {text_tokens.max():.0f}")
    print(f"  Mean: {text_tokens.mean():.1f}, Std: {text_tokens.std():.1f}")
    print(f"  Median: {np.median(text_tokens):.0f}")

    print(f"\nTotal tokens (audio + text):")
    print(f"  Min: {total_tokens.min():.1f}, Max: {total_tokens.max():.1f}")
    print(f"  Mean: {total_tokens.mean():.1f}, Std: {total_tokens.std():.1f}")
    print(f"  Median: {np.median(total_tokens):.1f}")

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"\nTotal token percentiles:")
    for p in percentiles:
        val = np.percentile(total_tokens, p)
        print(f"  P{p}: {val:.0f}")


def print_bucket_distribution(
    total_tokens: np.ndarray,
    bucket_bins: list[int],
    batch_sizes: list[int],
):
    """Print how samples are distributed across buckets."""
    print("\n" + "=" * 60)
    print("BUCKET DISTRIBUTION")
    print("=" * 60)

    prev_bin = 0
    for i, (bin_max, batch_size) in enumerate(zip(bucket_bins, batch_sizes)):
        count = np.sum((total_tokens > prev_bin) & (total_tokens <= bin_max))
        pct = 100 * count / len(total_tokens)
        print(f"  Bucket {i + 1}: ({prev_bin}, {bin_max}] tokens -> batch_size={batch_size}, samples={count} ({pct:.1f}%)")
        prev_bin = bin_max


def print_yaml_config(
    bucket_bins: list[int],
    batch_sizes: list[int],
    total_tokens: np.ndarray,
):
    """Print YAML config ready to paste into salm.yaml."""
    min_tokens = int(np.percentile(total_tokens, 1))
    max_tokens = int(np.percentile(total_tokens, 99))

    print("\n" + "=" * 60)
    print("YAML CONFIG (paste into salm.yaml data.train_ds)")
    print("=" * 60)
    print(f"""
use_bucketing: true
use_multimodal_sampling: true
min_tokens: {min_tokens}
max_tokens: {max_tokens}
bucket_duration_bins: {bucket_bins}
bucket_batch_size: {batch_sizes}
num_buckets: {len(bucket_bins)}
bucket_buffer_size: 5000
""")


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Loading Lhotse shar dataset from {args.shar_path}")
    cuts = load_lhotse_shar(args.shar_path, args.num_samples)
    print(f"Loaded {len(cuts)} cuts")

    audio_tokens, text_tokens, total_tokens = calculate_token_counts(
        cuts,
        tokenizer,
        args.token_equivalent_duration,
    )

    print_statistics(audio_tokens, text_tokens, total_tokens)

    bucket_bins = estimate_bucket_bins(total_tokens, args.num_buckets)

    median_tokens = np.median(total_tokens)
    batch_sizes = estimate_batch_sizes(
        bucket_bins,
        median_tokens,
        args.base_batch_size,
        args.base_memory_usage,
        args.min_batch_size,
        args.safety_factor,
    )

    print_bucket_distribution(total_tokens, bucket_bins, batch_sizes)
    print_yaml_config(bucket_bins, batch_sizes, total_tokens)

    print("=" * 60)
    print("NOTE: These are estimates. Monitor GPU memory during training")
    print("and adjust bucket_batch_size as needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
