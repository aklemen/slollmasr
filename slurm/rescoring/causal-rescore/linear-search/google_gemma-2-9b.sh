export LLM_NAME="google/gemma-2-9b"
export TOKENIZER_NAME=$LLM_NAME

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=causal-rescore-search-gemma-2-9b ../causal-rescore.sbatch