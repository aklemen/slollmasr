#!/bin/bash

INPUT_DIR="/shared/home/anton.klemen/manifests/artur/with-context"
OUTPUT_DIR="/shared/home/anton.klemen/manifests/artur/no-context"

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" \( -name "*.nemo" -o -name "*.nemo.filtered" \) | while read ORIGINAL_MANIFEST_FILE; do
  RELATIVE_PATH="${ORIGINAL_MANIFEST_FILE#$INPUT_DIR/}"
  NEW_MANIFEST_FILE="$OUTPUT_DIR/$RELATIVE_PATH"
  mkdir -p "$(dirname "$NEW_MANIFEST_FILE")"

  jq -c 'del(.context)' "$ORIGINAL_MANIFEST_FILE" > "$NEW_MANIFEST_FILE"

  echo "'context' field removed."
  echo "File available here: $NEW_MANIFEST_FILE."

done