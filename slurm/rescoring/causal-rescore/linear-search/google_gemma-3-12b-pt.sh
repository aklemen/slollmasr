export LLM_NAME="google/gemma-3-12b-pt"
export TOKENIZER_NAME=$LLM_NAME

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=causal-rescore-search-gemma-3-12b-pt ../causal-rescore.sbatch