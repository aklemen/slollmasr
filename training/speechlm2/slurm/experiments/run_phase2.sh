#!/bin/bash
# Phase 2: Encoder Unfreezing
# Run 02 - test whether unfreezing ASR encoder improves performance

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 2: Encoder Unfreezing ==="
echo "Submitting run02_encoder_unfreeze..."

source "$SCRIPT_DIR/configs/run02_encoder_unfreeze.env"

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

echo "Phase 2 submitted. Monitor progress in WandB project 'salm-ablations', group '$WANDB_GROUP'"
echo ""
echo "NEXT STEPS:"
echo "1. Wait for run01 and run02 to complete"
echo "2. Compare val_acc in WandB to determine best FREEZE_ENCODER setting"
echo "3. Update configs/run03_*.env, run04_*.env, run05_*.env with best FREEZE_ENCODER value"
echo "4. Run ./run_phase3.sh"
