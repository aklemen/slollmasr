export LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="task-activating-gec"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../prompting.sbatch