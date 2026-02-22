#!/usr/bin/env python3
"""
Find utterances with minimum and maximum total token counts in the artur clean train dataset.
Total tokens = context_tokens (chat format) + transcription_tokens + audio_tokens (duration/token_equivalent_duration)
"""
import json
from transformers import AutoTokenizer

manifest_path = "/shared/workspace/lpt-llm/datasets/artur/v1.0/nemo/clean/train.nemo"
token_equivalent_duration = 0.16  # 4x subsampling
prompt_text = "Prepiši govor v slovensko besedilo:"

# Tokenize prompt with both tokenizers
print("Loading tokenizers...")
gams_tokenizer = AutoTokenizer.from_pretrained("cjvt/GaMS-9B")
slovenian_gpt_tokenizer = AutoTokenizer.from_pretrained("aklemen/SlovenianGPT")

# Chat formats (tokenizers don't have chat_template set, so we apply manually)
# GaMS-9B uses gemma format, SlovenianGPT uses mistral format
gams_chat_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
slovenian_chat_prompt = f"[INST] {prompt_text} [/INST]"

# Context tokens (chat-formatted prompt) - constant for all samples
gams_context_tokens = len(gams_tokenizer.encode(gams_chat_prompt))
slovenian_context_tokens = len(slovenian_gpt_tokenizer.encode(slovenian_chat_prompt))

print(f"\n=== CONTEXT TOKENS (constant) ===")
print(f"GaMS-9B (gemma): {gams_context_tokens} tokens")
print(f"SlovenianGPT (mistral): {slovenian_context_tokens} tokens")

# Load samples and calculate total tokens
print(f"\nLoading manifest from: {manifest_path}")
samples = []
with open(manifest_path, 'rt') as f:
    for line in f:
        samples.append(json.loads(line))
print(f"Loaded {len(samples)} samples")

print("Calculating total tokens for each sample...")
for i, sample in enumerate(samples):
    text = sample.get('text', '')
    duration = sample.get('duration', 0)

    audio_tokens = duration / token_equivalent_duration

    gams_transcription_tokens = len(gams_tokenizer.encode(text))
    slovenian_transcription_tokens = len(slovenian_gpt_tokenizer.encode(text))

    sample['audio_tokens'] = audio_tokens
    sample['gams_transcription_tokens'] = gams_transcription_tokens
    sample['slovenian_transcription_tokens'] = slovenian_transcription_tokens
    sample['gams_total'] = gams_context_tokens + gams_transcription_tokens + audio_tokens
    sample['slovenian_total'] = slovenian_context_tokens + slovenian_transcription_tokens + audio_tokens

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1} samples...")

print(f"Finished processing {len(samples)} samples\n")


def print_sample_details(sample, model_name, context_tokens, transcription_key, total_key):
    print(f"  Duration: {sample['duration']:.2f}s")
    print(f"  Text: {sample['text']}")
    print(f"  Audio: {sample.get('audio_filepath', 'N/A')}")
    print(f"  Breakdown:")
    print(f"    Context tokens: {context_tokens}")
    print(f"    Transcription tokens: {sample[transcription_key]}")
    print(f"    Audio tokens: {sample['audio_tokens']:.1f}")
    print(f"    TOTAL: {sample[total_key]:.1f}")


# GaMS-9B results
samples_gams = sorted(samples, key=lambda x: x['gams_total'])
print("=" * 60)
print("GaMS-9B (gemma format)")
print("=" * 60)
print("\nMINIMUM TOTAL TOKENS:")
print_sample_details(samples_gams[0], "GaMS-9B", gams_context_tokens, 'gams_transcription_tokens', 'gams_total')
print("\nMAXIMUM TOTAL TOKENS:")
print_sample_details(samples_gams[-1], "GaMS-9B", gams_context_tokens, 'gams_transcription_tokens', 'gams_total')
print("\nDISTRIBUTION:")
print(f"  P5:  {samples_gams[int(len(samples) * 0.05)]['gams_total']:.1f} tokens")
print(f"  P50: {samples_gams[int(len(samples) * 0.50)]['gams_total']:.1f} tokens")
print(f"  P95: {samples_gams[int(len(samples) * 0.95)]['gams_total']:.1f} tokens")

# SlovenianGPT results
samples_slovenian = sorted(samples, key=lambda x: x['slovenian_total'])
print("\n" + "=" * 60)
print("SlovenianGPT (mistral format)")
print("=" * 60)
print("\nMINIMUM TOTAL TOKENS:")
print_sample_details(samples_slovenian[0], "SlovenianGPT", slovenian_context_tokens, 'slovenian_transcription_tokens',
                     'slovenian_total')
print("\nMAXIMUM TOTAL TOKENS:")
print_sample_details(samples_slovenian[-1], "SlovenianGPT", slovenian_context_tokens, 'slovenian_transcription_tokens',
                     'slovenian_total')
print("\nDISTRIBUTION:")
print(f"  P5:  {samples_slovenian[int(len(samples) * 0.05)]['slovenian_total']:.1f} tokens")
print(f"  P50: {samples_slovenian[int(len(samples) * 0.50)]['slovenian_total']:.1f} tokens")
print(f"  P95: {samples_slovenian[int(len(samples) * 0.95)]['slovenian_total']:.1f} tokens")
