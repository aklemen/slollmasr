#!/bin/bash
export LLM_NAME="cjvt/GaMS-9B"
export PROMPT_FORMAT="gemma"

export LORA_RANK=128
export LORA_ALPHA=128

# Bucketing config (from estimate_token_bins.py with 60% base memory usage)
export MIN_TOKENS=6
export MAX_TOKENS=95
export BUCKET_DURATION_BINS="[11,14,18,22,26,32,39,48,63,134]"
export BUCKET_BATCH_SIZE="[7,5,4,3,3,2,2,1,1,1]"

# Gradient accumulation (4 gives ~40 avg effective batch, max 112)
export GRADIENT_ACCUMULATION_STEPS=4

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=A100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch
