export LLM_NAME="google/gemma-2-27b"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE="32"

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=causal-rescore-search-gemma-2-27b ../causal-rescore.sbatch