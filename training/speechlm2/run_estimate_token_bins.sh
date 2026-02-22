#!/bin/bash
# Estimate token bins for SALM training using the actual dataset and tokenizer.
# This produces accurate bucket bins for use with oomptimizer and training.
#
# Usage: ./run_estimate_token_bins.sh <config_name> [num_samples]
#   config_name: slovenian_gpt | gams_9b
#   num_samples: Number of examples to sample (default: 10000, use -1 for all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to NeMo's estimate_token_bins.py script
ESTIMATE_SCRIPT="${ESTIMATE_SCRIPT:-/opt/NeMo/scripts/speech_llm/estimate_token_bins.py}"

# Check if script exists
if [[ ! -f "$ESTIMATE_SCRIPT" ]]; then
    echo "ERROR: estimate_token_bins.py not found at: $ESTIMATE_SCRIPT"
    echo "Set ESTIMATE_SCRIPT environment variable to the correct path."
    echo "Example: ESTIMATE_SCRIPT=/path/to/estimate_token_bins.py ./run_estimate_token_bins.sh gams_9b"
    exit 1
fi

# Parse config name
CONFIG_NAME="${1:-}"
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name> [num_samples]"
    echo "  config_name: slovenian_gpt | gams_9b"
    echo "  num_samples: Number of samples (default: 10000, use -1 for all)"
    exit 1
fi

case "$CONFIG_NAME" in
    slovenian_gpt)
        TOKENIZER="aklemen/SlovenianGPT"
        PROMPT_FORMAT="mistral"
        CONFIG_FILE="salm_slovenian_gpt.yaml"
        ;;
    gams_9b)
        TOKENIZER="cjvt/GaMS-9B"
        PROMPT_FORMAT="gemma"
        CONFIG_FILE="salm_gams_9b.yaml"
        ;;
    *)
        echo "Unknown config: $CONFIG_NAME"
        echo "Available: slovenian_gpt, gams_9b"
        exit 1
        ;;
esac

# Configuration
INPUT_CONFIG="${SCRIPT_DIR}/estimate_token_bins_input.yaml"
NUM_BUCKETS=10
NUM_SAMPLES="${2:-10000}"

# token_equivalent_duration from config
TOKEN_EQUIV_DURATION=0.32

echo "=============================================="
echo "Estimating token bins for SALM training"
echo "=============================================="
echo "Config: $CONFIG_NAME"
echo "Input config: $INPUT_CONFIG"
echo "Tokenizer: $TOKENIZER"
echo "Prompt format: $PROMPT_FORMAT"
echo "Number of buckets: $NUM_BUCKETS"
echo "Number of samples: $NUM_SAMPLES"
echo "Token equivalent duration: $TOKEN_EQUIV_DURATION"
echo "=============================================="
echo ""

python "$ESTIMATE_SCRIPT" \
    "$INPUT_CONFIG" \
    --tokenizer "$TOKENIZER" \
    --prompt-format "$PROMPT_FORMAT" \
    --buckets "$NUM_BUCKETS" \
    --measure-total-length True \
    --num_examples "$NUM_SAMPLES"

echo ""
echo "=============================================="
echo "Done! Use the bucket_duration_bins above in:"
echo "  - run_oomptimizer.sh (--buckets argument)"
echo "  - ${CONFIG_FILE} (data.train_ds.bucket_duration_bins)"
echo "=============================================="
