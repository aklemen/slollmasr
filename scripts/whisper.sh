#!/bin/bash

TIMESTAMP=$(date +%Y%m%d%H%M%S)

if ! pip show whisper; then
  echo "Installing whisper..."
  /slollmasr/scripts/install-whisper.sh
fi

python /slollmasr/whisper_transcribe.py \
  --manifest_file_path "/dataset/artur/v1.0/nemo/test.nemo" \
  --beams_file_path "/testing/beams_$TIMESTAMP.tsv" \
  --beam_width 5