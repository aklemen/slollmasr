export LLM_NAME="utter-project/EuroLLM-9B"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=64

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=causal-rescore-search-eurollm-9b ../causal-rescore.sbatch