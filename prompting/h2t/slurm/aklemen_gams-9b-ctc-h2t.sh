export LLM_NAME="aklemen/gams-9b-ctc-h2t"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="h2t-mapping"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch