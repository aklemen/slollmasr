export SHOULD_LINEAR_SEARCH_ARTUR_DEV=true

export LLM_NAME="utter-project/EuroLLM-9B"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../rescoring.sbatch