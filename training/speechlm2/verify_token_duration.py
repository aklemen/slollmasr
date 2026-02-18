#!/usr/bin/env python3
"""
Verify token_equivalent_duration by measuring actual encoder output rate.

Runs audio samples through the ASR encoder + modality adapter and compares
the output sequence length to the audio duration.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify token_equivalent_duration for SpeechLM2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to NeMo manifest JSON file",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="aklemen/slovene-conformer-ctc",
        help="ASR encoder model name or path",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to salm.yaml for modality adapter config (default: same directory as this script)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Expected audio sample rate",
    )
    return parser.parse_args()


def load_manifest(manifest_path: str, num_samples: int) -> list[dict]:
    """Load entries from NeMo manifest file."""
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            entries.append(entry)

    if num_samples > 0 and len(entries) > num_samples:
        entries = random.sample(entries, num_samples)

    return entries


def load_asr_encoder(model_name: str, device: str):
    """Load ASR encoder from NeMo."""
    from nemo.collections.asr.models import EncDecCTCModel

    model = EncDecCTCModel.from_pretrained(model_name, map_location=device)
    model.eval()
    model.freeze()
    return model


def create_modality_adapter(config_path: str, device: str):
    """Create modality adapter from config."""
    from hydra.utils import instantiate

    config = OmegaConf.load(config_path)
    adapter_config = config.model.perception.modality_adapter

    adapter = instantiate(adapter_config)
    adapter = adapter.to(device)
    adapter.eval()
    return adapter


def load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Load and resample audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def main():
    args = parse_args()

    if args.config_path is None:
        args.config_path = str(Path(__file__).parent / "salm.yaml")

    print(f"Loading manifest from {args.manifest}")
    entries = load_manifest(args.manifest, args.num_samples)
    print(f"Loaded {len(entries)} samples")

    print(f"Loading ASR encoder: {args.asr_model}")
    asr_model = load_asr_encoder(args.asr_model, args.device)

    print(f"Creating modality adapter from {args.config_path}")
    adapter = create_modality_adapter(args.config_path, args.device)

    durations = []
    output_lengths = []
    token_durations = []

    print("\nProcessing samples...")
    for entry in tqdm(entries):
        audio_path = entry["audio_filepath"]
        duration = entry["duration"]

        try:
            audio = load_audio(audio_path, args.sample_rate).to(args.device)
            audio_len = torch.tensor([audio.shape[0]], device=args.device)

            with torch.no_grad():
                encoded, encoded_len = asr_model.encoder(
                    audio_signal=audio.unsqueeze(0),
                    length=audio_len,
                )
                adapter_out, adapter_len = adapter(
                    audio_signal=encoded,
                    length=encoded_len,
                )

            out_len = adapter_len.item()
            token_dur = duration / out_len

            durations.append(duration)
            output_lengths.append(out_len)
            token_durations.append(token_dur)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    durations = np.array(durations)
    output_lengths = np.array(output_lengths)
    token_durations = np.array(token_durations)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAnalyzed {len(durations)} samples")
    print(f"Audio durations: {durations.min():.1f}s - {durations.max():.1f}s (mean: {durations.mean():.1f}s)")
    print(f"Output lengths: {output_lengths.min()} - {output_lengths.max()} tokens (mean: {output_lengths.mean():.0f})")
    print(f"\nActual token_equivalent_duration: {token_durations.mean():.4f}s ± {token_durations.std():.4f}s")

    recommended = round(token_durations.mean(), 2)
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION: token_equivalent_duration = {recommended}")
    print(f"{'=' * 60}")

    tokens_per_second = 1 / token_durations.mean()
    print(f"\nThis corresponds to ~{tokens_per_second:.1f} audio tokens per second")


if __name__ == "__main__":
    main()
