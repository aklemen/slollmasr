export LLM_NAME="google/gemma-2-9b-it"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="zero-shot-selection"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch