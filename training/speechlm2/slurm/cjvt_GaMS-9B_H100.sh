#!/bin/bash
export LLM_NAME="cjvt/GaMS-9B"
export PROMPT_FORMAT="gemma"

export LORA_RANK=128
export LORA_ALPHA=128

# Bucketing config (scaled ~10x for H100 80GB from A100 40GB estimates)
export MIN_TOKENS=6
export MAX_TOKENS=95
export BUCKET_DURATION_BINS="[11,14,18,22,26,32,39,48,63,134]"
export BUCKET_BATCH_SIZE="[72,50,42,30,30,20,20,8,8,8]"

# Gradient accumulation (1 gives ~100 avg effective batch, max 288)
export GRADIENT_ACCUMULATION_STEPS=1

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=256G \
  train_speechlm2.sbatch
