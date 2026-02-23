#!/bin/bash
export CONFIG_NAME="sloveniangpt_baseline"

JOB_NAME="train-salm-$(basename "$CONFIG_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch
