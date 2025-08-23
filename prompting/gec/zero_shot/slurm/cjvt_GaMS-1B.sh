export LLM_NAME="cjvt/GaMS-1B-Chat"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="zero-shot-gec"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch