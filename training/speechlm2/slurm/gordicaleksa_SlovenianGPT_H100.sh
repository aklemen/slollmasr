#!/bin/bash
export LLM_NAME="aklemen/SlovenianGPT"
export PROMPT_FORMAT="mistral"

export LORA_RANK=128
export LORA_ALPHA=128

export MIN_TOKENS=1
export MAX_TOKENS=1024
export BUCKET_DURATION_BINS="[37,40,44,49,54,60,68,79,96,188]"
export BUCKET_BATCH_SIZE="[24,24,24,18,18,18,12,12,6,6]"

export GRADIENT_ACCUMULATION_STEPS=1

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch
