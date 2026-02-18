#!/usr/bin/env python3
"""
Estimate token-based bucketing configuration for SpeechLM2.

Analyzes a Lhotse shar dataset to determine optimal bucket_duration_bins
and bucket_batch_size for efficient training. Supports multiple tokenizers
in a single run.

Token calculation matches NeMo's MultimodalFixedBucketBatchSizeConstraint2D:
  total_tokens = ceil(duration / token_equivalent_duration) + len(input_ids)

Where input_ids is the full tokenized prompt including:
  - BOS token
  - User turn: <start_of_turn>user\n{context} {audio_locator}<end_of_turn>\n
  - Assistant turn: <start_of_turn>model\n{transcript}<end_of_turn>
  - EOS token
"""

import argparse
import math

import numpy as np
from tqdm import tqdm


# Prompt templates matching NeMo's implementations
PROMPT_TEMPLATES = {
    "gemma": {
        "user_prefix": "<start_of_turn>user\n",
        "user_suffix": "<end_of_turn>\n<start_of_turn>model\n",
        "assistant_suffix": "<end_of_turn>",
        "insert_bos": True,
        "insert_eos": True,
    },
    "mistral": {
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST] ",
        "assistant_suffix": "",
        "insert_bos": True,
        "insert_eos": True,
    },
    "llama": {
        "user_prefix": "[INST] ",
        "user_suffix": " [/INST] ",
        "assistant_suffix": "",
        "insert_bos": True,
        "insert_eos": True,
    },
    "none": {
        "user_prefix": "",
        "user_suffix": "",
        "assistant_suffix": "",
        "insert_bos": False,
        "insert_eos": False,
    },
}


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
        action="append",
        required=True,
        dest="tokenizers",
        help="HuggingFace tokenizer name or path (can specify multiple times)",
    )
    parser.add_argument(
        "--token-equivalent-duration",
        type=float,
        required=True,
        help="Seconds per audio token (get from verify_token_duration.py)",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        action="append",
        dest="prompt_formats",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt format per tokenizer (can specify multiple times, must match salm.yaml)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Context string prepended to audio (e.g., 'Transkribiraj naslednji posnetek:')",
    )
    parser.add_argument(
        "--audio-locator-tag",
        type=str,
        default="<audio>",
        help="Audio placeholder token used in prompts",
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
        action="append",
        dest="base_memory_usages",
        help="GPU memory fraction used at base batch size (0-1). Can specify multiple times, one per tokenizer.",
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Fixed max_tokens value (use large value to avoid filtering samples)",
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


def extract_texts_and_durations(cuts: list) -> tuple[list[str], np.ndarray]:
    """Extract texts and durations from cuts (do this once)."""
    texts = []
    durations = []

    for cut in tqdm(cuts, desc="Extracting metadata"):
        durations.append(cut.duration)
        text = ""
        if cut.supervisions:
            text = cut.supervisions[0].text or ""
        texts.append(text)

    return texts, np.array(durations)


def build_formatted_prompt(
    context: str,
    audio_locator_tag: str,
    transcript: str,
    prompt_format: str,
) -> str:
    """
    Build the full formatted prompt string as NeMo does.

    For gemma format with context "Transkribiraj naslednji posnetek:" and transcript "hello":
    <start_of_turn>user
    Transkribiraj naslednji posnetek: <audio><end_of_turn>
    <start_of_turn>model
    hello<end_of_turn>

    Note: BOS/EOS are handled separately during tokenization.
    """
    template = PROMPT_TEMPLATES[prompt_format]

    # Build user message: context + space + audio_locator (if context exists)
    if context:
        user_message = f"{context} {audio_locator_tag}"
    else:
        user_message = audio_locator_tag

    # Build full prompt (without BOS/EOS - those are added by tokenizer)
    prompt = (
        f"{template['user_prefix']}"
        f"{user_message}"
        f"{template['user_suffix']}"
        f"{transcript}"
        f"{template['assistant_suffix']}"
    )

    return prompt


def calculate_token_counts(
    texts: list[str],
    durations: np.ndarray,
    tokenizer,
    token_equivalent_duration: float,
    prompt_format: str,
    context: str,
    audio_locator_tag: str,
    tokenizer_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Calculate audio, text, and total token counts matching NeMo's calculation.

    Returns:
        audio_tokens: ceil(duration / token_equivalent_duration) for each sample
        text_tokens: len(tokenize(full_formatted_prompt)) for each sample
        total_tokens: audio_tokens + text_tokens
        prompt_overhead: fixed overhead from prompt template (without transcript)
    """
    template = PROMPT_TEMPLATES[prompt_format]

    # Calculate audio tokens using ceil (matching NeMo)
    audio_tokens = np.ceil(durations / token_equivalent_duration)

    # Calculate prompt overhead (fixed part without transcript)
    # This helps users understand how much the prompt adds
    empty_prompt = build_formatted_prompt(context, audio_locator_tag, "", prompt_format)
    overhead_tokens = tokenizer.encode(empty_prompt, add_special_tokens=False)

    # Add BOS/EOS if template requires it
    prompt_overhead = len(overhead_tokens)
    if template["insert_bos"] and tokenizer.bos_token_id is not None:
        prompt_overhead += 1
    if template["insert_eos"] and tokenizer.eos_token_id is not None:
        prompt_overhead += 1

    # Tokenize each full prompt
    text_tokens_list = []
    for text in tqdm(texts, desc=f"Tokenizing ({tokenizer_name})"):
        full_prompt = build_formatted_prompt(context, audio_locator_tag, text, prompt_format)

        # Tokenize without special tokens first, then add them
        tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
        num_tokens = len(tokens)

        # Add BOS/EOS counts
        if template["insert_bos"] and tokenizer.bos_token_id is not None:
            num_tokens += 1
        if template["insert_eos"] and tokenizer.eos_token_id is not None:
            num_tokens += 1

        text_tokens_list.append(num_tokens)

    text_tokens = np.array(text_tokens_list)
    total_tokens = audio_tokens + text_tokens

    return audio_tokens, text_tokens, total_tokens, prompt_overhead


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
    tokenizer_name: str,
    audio_tokens: np.ndarray,
    text_tokens: np.ndarray,
    total_tokens: np.ndarray,
    prompt_overhead: int,
    prompt_format: str,
):
    """Print dataset statistics."""
    print(f"\nSamples analyzed: {len(total_tokens)}")

    print(f"\nPrompt format: {prompt_format}")
    print(f"  Fixed overhead (template + context + audio_locator + BOS/EOS): {prompt_overhead} tokens")

    print(f"\nAudio tokens (ceil(duration / token_equivalent_duration)):")
    print(f"  Min: {audio_tokens.min():.0f}, Max: {audio_tokens.max():.0f}")
    print(f"  Mean: {audio_tokens.mean():.1f}, Std: {audio_tokens.std():.1f}")
    print(f"  Median: {np.median(audio_tokens):.0f}")

    print(f"\nText tokens (full formatted prompt with transcript):")
    print(f"  Min: {text_tokens.min():.0f}, Max: {text_tokens.max():.0f}")
    print(f"  Mean: {text_tokens.mean():.1f}, Std: {text_tokens.std():.1f}")
    print(f"  Median: {np.median(text_tokens):.0f}")

    # Show transcript-only tokens (text_tokens - prompt_overhead)
    transcript_only = text_tokens - prompt_overhead
    print(f"\nTranscript-only tokens (text_tokens - overhead):")
    print(f"  Min: {transcript_only.min():.0f}, Max: {transcript_only.max():.0f}")
    print(f"  Mean: {transcript_only.mean():.1f}, Median: {np.median(transcript_only):.0f}")

    print(f"\nTotal tokens (audio + text):")
    print(f"  Min: {total_tokens.min():.0f}, Max: {total_tokens.max():.0f}")
    print(f"  Mean: {total_tokens.mean():.1f}, Std: {total_tokens.std():.1f}")
    print(f"  Median: {np.median(total_tokens):.0f}")

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
    print("\nBUCKET DISTRIBUTION:")

    prev_bin = 0
    for i, (bin_max, batch_size) in enumerate(zip(bucket_bins, batch_sizes)):
        count = np.sum((total_tokens > prev_bin) & (total_tokens <= bin_max))
        pct = 100 * count / len(total_tokens)
        print(f"  Bucket {i + 1}: ({prev_bin}, {bin_max}] tokens -> batch_size={batch_size}, samples={count} ({pct:.1f}%)")
        prev_bin = bin_max

    # Check for samples exceeding max bin
    over_max = np.sum(total_tokens > bucket_bins[-1])
    if over_max > 0:
        pct = 100 * over_max / len(total_tokens)
        print(f"  WARNING: {over_max} samples ({pct:.1f}%) exceed max bin ({bucket_bins[-1]})")


def print_yaml_config(
    tokenizer_name: str,
    bucket_bins: list[int],
    batch_sizes: list[int],
    total_tokens: np.ndarray,
    max_tokens: int,
):
    """Print YAML config ready to paste into salm.yaml."""
    min_tokens = int(np.percentile(total_tokens, 1))

    print(f"\nYAML CONFIG for {tokenizer_name}:")
    print(f"# Paste into salm.yaml data.train_ds or use in SLURM script")
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


def get_model_short_name(tokenizer_name: str) -> str:
    """Extract short name from tokenizer path."""
    return tokenizer_name.split("/")[-1]


def main():
    args = parse_args()

    # Handle base_memory_usages - default to 0.6 if not specified
    if args.base_memory_usages is None:
        args.base_memory_usages = [0.6]

    # Extend memory usages to match number of tokenizers
    while len(args.base_memory_usages) < len(args.tokenizers):
        args.base_memory_usages.append(args.base_memory_usages[-1])

    # Handle prompt_formats - default to "none" if not specified
    if args.prompt_formats is None:
        args.prompt_formats = ["none"]

    # Extend prompt_formats to match number of tokenizers
    while len(args.prompt_formats) < len(args.tokenizers):
        args.prompt_formats.append(args.prompt_formats[-1])

    print(f"Loading Lhotse shar dataset from {args.shar_path}")
    cuts = load_lhotse_shar(args.shar_path, args.num_samples)
    print(f"Loaded {len(cuts)} cuts")

    # Extract texts and durations once
    texts, durations = extract_texts_and_durations(cuts)

    # Free memory from cuts
    del cuts

    print(f"\nConfiguration:")
    print(f"  token_equivalent_duration: {args.token_equivalent_duration}")
    print(f"  context: '{args.context}'")
    print(f"  audio_locator_tag: '{args.audio_locator_tag}'")

    # Process each tokenizer
    for tokenizer_name, base_memory_usage, prompt_format in zip(
        args.tokenizers, args.base_memory_usages, args.prompt_formats
    ):
        short_name = get_model_short_name(tokenizer_name)

        print("\n" + "=" * 70)
        print(f"RESULTS FOR: {tokenizer_name}")
        print(f"Base memory usage: {base_memory_usage:.0%}")
        print(f"Prompt format: {prompt_format}")
        print("=" * 70)

        print(f"\nLoading tokenizer: {tokenizer_name}")
        tokenizer = load_tokenizer(tokenizer_name)

        audio_tokens, text_tokens, total_tokens, prompt_overhead = calculate_token_counts(
            texts,
            durations,
            tokenizer,
            args.token_equivalent_duration,
            prompt_format,
            args.context,
            args.audio_locator_tag,
            short_name,
        )

        print_statistics(short_name, audio_tokens, text_tokens, total_tokens, prompt_overhead, prompt_format)

        bucket_bins = estimate_bucket_bins(total_tokens, args.num_buckets)

        median_tokens = np.median(total_tokens)
        batch_sizes = estimate_batch_sizes(
            bucket_bins,
            median_tokens,
            args.base_batch_size,
            base_memory_usage,
            args.min_batch_size,
            args.safety_factor,
        )

        print_bucket_distribution(total_tokens, bucket_bins, batch_sizes)
        print_yaml_config(short_name, bucket_bins, batch_sizes, total_tokens, args.max_tokens)

        # Free tokenizer memory
        del tokenizer

    print("\n" + "=" * 70)
    print("NOTE: These are estimates. Monitor GPU memory during training")
    print("and adjust bucket_batch_size as needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
