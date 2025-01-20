#!/bin/bash

PROMPT="What is the transcription of this audio?"

LIBRISPEECH_NEMO_DIR="$HOME/datasets/librispeech"
OUTPUT_DIR="$HOME/manifests/librispeech/what-is-the-transcription-of-this-audio"

CONTEXT_KEY="context"
ANSWER_KEY="answer"

mkdir -p "$OUTPUT_DIR"

find "$LIBRISPEECH_NEMO_DIR" \( -name "*.json" -o -name "*.nemo.filtered" \) | while read ORIGINAL_MANIFEST_FILE; do
  RELATIVE_PATH="${ORIGINAL_MANIFEST_FILE#$LIBRISPEECH_NEMO_DIR/}"
  NEW_MANIFEST_FILE="$OUTPUT_DIR/$RELATIVE_PATH"
  mkdir -p "$(dirname "$NEW_MANIFEST_FILE")"

  jq -c --arg CONTEXT_KEY "$CONTEXT_KEY" --arg PROMPT "$PROMPT" --arg ANSWER_KEY "$ANSWER_KEY" '
  . + {($CONTEXT_KEY): $PROMPT} |
  . + {($ANSWER_KEY): .text} |
  del(.text)' "$ORIGINAL_MANIFEST_FILE" > "$NEW_MANIFEST_FILE"
  
  echo "'$CONTEXT_KEY' field added and 'text' field renamed to '$ANSWER_KEY'."
  echo "File available here: $NEW_MANIFEST_FILE."

done