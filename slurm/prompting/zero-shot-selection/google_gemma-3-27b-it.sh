export LLM_NAME="google/gemma-3-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=zero-shot-selection-gemma-3-27b-it zero-shot-selection.sbatch