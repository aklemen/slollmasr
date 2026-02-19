#!/bin/bash
# Phase 1: Baseline
# Run 01 - frozen encoder, standard adapter, baseline LoRA

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1: Baseline ==="
echo "Submitting run01_baseline..."

source "$SCRIPT_DIR/configs/run01_baseline.env"

# Calculate token_equivalent_duration from subsampling factor
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
export GRADIENT_ACCUMULATION_STEPS=1

echo "Config:"
echo "  RUN_NAME=$RUN_NAME"
echo "  FREEZE_ENCODER=$FREEZE_ENCODER"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING"
echo "  ADAPTER_N_LAYERS=$ADAPTER_N_LAYERS"
echo "  ADAPTER_D_MODEL=$ADAPTER_D_MODEL"
echo "  LORA_RANK=$LORA_RANK"
echo "  LORA_ALPHA=$LORA_ALPHA"
echo "  LORA_TARGET_MODULES=$LORA_TARGET_MODULES"
echo "  LLM_NAME=$LLM_NAME"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"

sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"

echo "Phase 1 submitted. Monitor progress in WandB project 'salm-ablations', group '$WANDB_GROUP'"
