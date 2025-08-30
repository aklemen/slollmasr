export LLM_NAME="google/gemma-3-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="zero-shot-selection"
export BATCH_SIZE=8

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../prompting.sbatch