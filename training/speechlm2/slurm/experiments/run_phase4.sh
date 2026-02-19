#!/bin/bash
# Phase 4: LoRA Configuration Ablations (runs in parallel)
# Run 06 - lora_r=64
# Run 07 - lora_r=256
# Run 08 - target_modules=all-linear
#
# PREREQUISITE: Update FREEZE_ENCODER and adapter params in run06, run07, run08 configs
#               based on Phase 2-3 results

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 4: LoRA Configuration Ablations ==="
echo "Submitting runs 06, 07, 08 in parallel..."
echo ""

export GRADIENT_ACCUMULATION_STEPS=1

# Run 06: lora_r=64
echo "--- Run 06: lora_r=64 ---"
source "$SCRIPT_DIR/configs/run06_lora_r64.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING (should be set based on Phase 3)"
echo "  LORA_RANK=$LORA_RANK"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

# Run 07: lora_r=256
echo "--- Run 07: lora_r=256 ---"
source "$SCRIPT_DIR/configs/run07_lora_r256.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING (should be set based on Phase 3)"
echo "  LORA_RANK=$LORA_RANK"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

# Run 08: all-linear target modules
echo "--- Run 08: all-linear target modules ---"
source "$SCRIPT_DIR/configs/run08_all_linear.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING (should be set based on Phase 3)"
echo "  LORA_TARGET_MODULES=$LORA_TARGET_MODULES"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

echo "Phase 4 submitted (3 jobs in parallel). Monitor progress in WandB project 'salm-ablations', group '$WANDB_GROUP'"
echo ""
echo "NEXT STEPS:"
echo "1. Wait for runs 06, 07, 08 to complete"
echo "2. Compare val_acc to determine best LoRA config"
echo "3. Update configs/run09_sloveniangpt.env with all best params from Phase 2-4"
echo "4. Run ./run_phase5.sh"
