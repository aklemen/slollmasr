#!/bin/bash
# Phase 5: SlovenianGPT Transfer
# Run 09 - test best configuration with SlovenianGPT instead of GaMS-9B
#
# PREREQUISITE: Update ALL params in run09_sloveniangpt.env based on Phase 2-4 results

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 5: SlovenianGPT Transfer ==="
echo "Submitting run09_sloveniangpt..."

source "$SCRIPT_DIR/configs/run09_sloveniangpt.env"

# Calculate token_equivalent_duration from subsampling factor
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
export GRADIENT_ACCUMULATION_STEPS=1

echo "Config:"
echo "  RUN_NAME=$RUN_NAME"
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING (should be set based on Phase 3)"
echo "  ADAPTER_N_LAYERS=$ADAPTER_N_LAYERS (should be set based on Phase 3)"
echo "  ADAPTER_D_MODEL=$ADAPTER_D_MODEL (should be set based on Phase 3)"
echo "  LORA_RANK=$LORA_RANK (should be set based on Phase 4)"
echo "  LORA_ALPHA=$LORA_ALPHA"
echo "  LORA_TARGET_MODULES=$LORA_TARGET_MODULES (should be set based on Phase 4)"
echo "  LLM_NAME=$LLM_NAME"
echo "  PROMPT_FORMAT=$PROMPT_FORMAT"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"

sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"

echo "Phase 5 submitted. Monitor progress in WandB project 'salm-ablations', group '$WANDB_GROUP'"
echo ""
echo "FINAL STEPS:"
echo "1. Wait for run09 to complete"
echo "2. Convert best checkpoints to HuggingFace format:"
echo "   python convert_salm_ckpt_to_hf.py --ckpt_dir <path> --config_path <path>/logs/exp_config.yaml"
echo "3. Run final evaluation on all 6 test datasets"
echo "4. Compare greedy vs nucleus sampling on best models"
