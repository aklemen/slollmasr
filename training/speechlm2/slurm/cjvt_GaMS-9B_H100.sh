#!/bin/bash
export LLM_NAME="cjvt/GaMS-9B"
export PROMPT_FORMAT="gemma"

export LORA_RANK=128
export LORA_ALPHA=128

export MIN_TOKENS=1
export MAX_TOKENS=1024
export BUCKET_DURATION_BINS="[34,37,41,45,49,55,62,71,86,157]"
export BUCKET_BATCH_SIZE="[28,20,20,20,14,14,14,8,8,8]"

export GRADIENT_ACCUMULATION_STEPS=1

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch
