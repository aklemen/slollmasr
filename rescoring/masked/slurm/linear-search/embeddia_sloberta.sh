export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

export LLM_NAME="EMBEDDIA/sloberta"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="masked-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../rescoring.sbatch