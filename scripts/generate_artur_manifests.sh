#!/bin/bash

PROMPT="Kaj je transkript tega avdia?"

ARTUR_NEMO_DIR="/shared/workspace/lpt-llm/datasets/artur/v1.0/nemo"
OUTPUT_DIR="/shared/home/anton.klemen/manifests/artur/kaj-je-transkript-tega-avdia"

CONTEXT_KEY="context"
ANSWER_KEY="answer"

mkdir -p "$OUTPUT_DIR"

find "$ARTUR_NEMO_DIR" \( -name "*.nemo" -o -name "*.nemo.filtered" \) | while read ORIGINAL_MANIFEST_FILE; do
  RELATIVE_PATH="${ORIGINAL_MANIFEST_FILE#$ARTUR_NEMO_DIR/}"
  NEW_MANIFEST_FILE="$OUTPUT_DIR/$RELATIVE_PATH"
  mkdir -p "$(dirname "$NEW_MANIFEST_FILE")"

  jq -c --arg CONTEXT_KEY "$CONTEXT_KEY" --arg PROMPT "$PROMPT" --arg ANSWER_KEY "$ANSWER_KEY" '
  . + {($CONTEXT_KEY): $PROMPT} |
  . + {($ANSWER_KEY): .text} |
  del(.text)' "$ORIGINAL_MANIFEST_FILE" > "$NEW_MANIFEST_FILE"
  
  echo "'$CONTEXT_KEY' field added and 'text' field renamed to '$ANSWER_KEY'."
  echo "File available here: $NEW_MANIFEST_FILE."

done