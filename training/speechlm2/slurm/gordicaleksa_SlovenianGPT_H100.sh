#!/bin/bash
export LLM_NAME="aklemen/SlovenianGPT"
export PROMPT_FORMAT="mistral"

export LORA_RANK=128
export LORA_ALPHA=128

# Bucketing config (scaled ~2.5x for H100 80GB from A100 40GB estimates)
export MIN_TOKENS=7
export MAX_TOKENS=109
export BUCKET_DURATION_BINS="[13,16,20,25,30,36,44,55,72,164]"
export BUCKET_BATCH_SIZE="[20,15,12,10,7,7,5,5,2,2]"

# Gradient accumulation (2 gives ~60 avg effective batch, max 160)
export GRADIENT_ACCUMULATION_STEPS=2

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=256G \
  train_speechlm2.sbatch
