export LLM_NAME="EMBEDDIA/sloberta"
export TOKENIZER_NAME=$LLM_NAME

export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

sbatch --job-name=masked-rescore-search-sloberta ../masked-rescore.sbatch