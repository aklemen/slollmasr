export LLM_NAME="meta-llama/Llama-3.1-70B"
export TOKENIZER_NAME=$LLM_NAME

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=causal-rescore-search-llama-31-70b ../causal-rescore.sbatch