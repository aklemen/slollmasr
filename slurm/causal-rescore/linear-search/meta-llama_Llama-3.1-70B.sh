export LLM_NAME="meta-llama/Llama-3.1-70B"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE="32"

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch ../causal-rescore.sbatch