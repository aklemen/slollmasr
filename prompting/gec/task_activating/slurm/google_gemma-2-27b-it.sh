export LLM_NAME="google/gemma-2-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="task-activating-gec"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch