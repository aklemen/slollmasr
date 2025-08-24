export LLM_NAME="google/gemma-3-12b-it"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="zero-shot-gec"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../prompting.sbatch