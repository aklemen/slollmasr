export LLM_NAME="meta-llama/Llama-3.3-70B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="one-shot-gec"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch