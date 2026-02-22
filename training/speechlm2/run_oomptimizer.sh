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
        ;;
    gams_9b)
        CONFIG_PATH="${SCRIPT_DIR}/salm_gams_9b.yaml"
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

BUCKET_BINS="[2.254,3.182,4.068,4.983,5.942,7.002,8.22,9.686,11.738]"

echo "Running oomptimizer with config: $CONFIG_PATH"
echo "Bucket bins: $BUCKET_BINS"

python "$OOMPTIMIZER_PATH" \
    --module-name nemo.collections.speechlm2.SALM \
    --config-path "$CONFIG_PATH" \
    --buckets "$BUCKET_BINS"
