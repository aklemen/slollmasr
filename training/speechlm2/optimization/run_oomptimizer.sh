#!/bin/bash
# Run oomptimizer to find optimal batch sizes for SALM training.
# Usage: ./run_oomptimizer.sh <config_name>
#   config_name: slovenian_gpt | gams_9b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OOMPTIMIZER_PATH="${SCRIPT_DIR}/oomptimizer_wrapper.py"

CONFIG_NAME="${1:-}"
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name>"
    echo "  config_name: slovenian_gpt | gams_9b"
    exit 1
fi

case "$CONFIG_NAME" in
    slovenian_gpt)
        CONFIG_PATH="${SCRIPT_DIR}/salm_slovenian_gpt.yaml"
        # SlovenianGPT: ~5.11 tokens/sec, token_equivalent_duration ~0.196s
        # Bucket bins in total tokens (adjust based on estimate_token_bins.py output)
        BUCKET_BINS="[47,55,64,73,83,94,108,125,148]"
        ;;
    gams_9b)
        CONFIG_PATH="${SCRIPT_DIR}/salm_gams_9b.yaml"
        # GaMS-9B: ~4.08 tokens/sec, token_equivalent_duration ~0.245s
        # Bucket bins in total tokens (adjust based on estimate_token_bins.py output)
        BUCKET_BINS="[44,52,59,67,76,87,99,114,136]"
        ;;
    *)
        echo "Unknown config: $CONFIG_NAME"
        echo "Available: slovenian_gpt, gams_9b"
        exit 1
        ;;
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Running oomptimizer with config: $CONFIG_PATH"
echo "Bucket bins: $BUCKET_BINS"

python "$OOMPTIMIZER_PATH" \
    --module-name nemo.collections.speechlm2.SALM \
    --config-path "$CONFIG_PATH" \
    --buckets "$BUCKET_BINS"
