export LLM_NAME="google/gemma-2-9b"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE="64"

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch ../causal-rescore.sbatch