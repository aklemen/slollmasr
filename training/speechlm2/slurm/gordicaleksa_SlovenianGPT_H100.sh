#!/bin/bash
export LLM_NAME="aklemen/SlovenianGPT"
export PROMPT_FORMAT="mistral"

export LORA_RANK=128
export LORA_ALPHA=128

# Bucketing config (scaled ~10x for H100 80GB from A100 40GB estimates)
export MIN_TOKENS=7
export MAX_TOKENS=109
export BUCKET_DURATION_BINS="[13,16,20,25,30,36,44,55,72,164]"
export BUCKET_BATCH_SIZE="[85,62,50,42,30,30,20,20,8,8]"

# Gradient accumulation (1 gives ~120 avg effective batch, max 340)
export GRADIENT_ACCUMULATION_STEPS=1

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch
