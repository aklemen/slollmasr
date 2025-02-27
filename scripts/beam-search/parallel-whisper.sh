#!/bin/bash

TIMESTAMP=$(date +%Y%m%d%H%M%S)

if ! pip show whisper; then
  echo "Installing whisper..."
  /slollmasr/scripts/beam-search/install-whisper.sh
fi

RESULTS_DIR="/testing/parallel-whisper/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

python /slollmasr/parallel_whisper_transcribe.py \
  --manifest_file_path "/dataset/artur/v1.0/nemo/clean/train.nemo" \
  --results_dir_path "$RESULTS_DIR" \
  --beam_width 10 \
  --save_frequency 10