#!/bin/bash
# Phase 3: Adapter Architecture Ablations (runs in parallel)
# Run 03 - subsampling=4
# Run 04 - n_layers=4
# Run 05 - d_model=2048
#
# PREREQUISITE: Update FREEZE_ENCODER in run03, run04, run05 configs based on Phase 2 results

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 3: Adapter Architecture Ablations ==="
echo "Submitting runs 03, 04, 05 in parallel..."
echo ""

export GRADIENT_ACCUMULATION_STEPS=1

# Run 03: subsampling=4
echo "--- Run 03: subsampling=4 ---"
source "$SCRIPT_DIR/configs/run03_subsampling4.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_SUBSAMPLING=$ADAPTER_SUBSAMPLING"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

# Run 04: n_layers=4
echo "--- Run 04: n_layers=4 ---"
source "$SCRIPT_DIR/configs/run04_layers4.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_N_LAYERS=$ADAPTER_N_LAYERS"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

# Run 05: d_model=2048
echo "--- Run 05: d_model=2048 ---"
source "$SCRIPT_DIR/configs/run05_dmodel2048.env"
export TOKEN_EQUIVALENT_DURATION=$(echo "$ADAPTER_SUBSAMPLING * 0.04" | bc)
echo "  FREEZE_ENCODER=$FREEZE_ENCODER (should be set based on Phase 2)"
echo "  ADAPTER_D_MODEL=$ADAPTER_D_MODEL"
echo "  TOKEN_EQUIVALENT_DURATION=$TOKEN_EQUIVALENT_DURATION"
sbatch \
  --job-name="salm-$RUN_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  "$SCRIPT_DIR/../train_speechlm2.sbatch"
echo ""

echo "Phase 3 submitted (3 jobs in parallel). Monitor progress in WandB project 'salm-ablations', group '$WANDB_GROUP'"
echo ""
echo "NEXT STEPS:"
echo "1. Wait for runs 03, 04, 05 to complete"
echo "2. Compare val_acc to run01/02 baseline to determine best adapter config"
echo "3. Update configs/run06_*.env, run07_*.env, run08_*.env with best adapter params"
echo "4. Run ./run_phase4.sh"
