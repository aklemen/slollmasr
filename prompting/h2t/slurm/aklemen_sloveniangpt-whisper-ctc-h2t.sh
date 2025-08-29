export LLM_NAME="aklemen/sloveniangpt-whisper-ctc-h2t-128-64"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="h2t-mapping"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../prompting.sbatch