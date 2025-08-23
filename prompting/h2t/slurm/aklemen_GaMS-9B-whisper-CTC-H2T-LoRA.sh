export LLM_NAME="aklemen/GaMS-9B-whisper-CTC-H2T-LoRA"
export TOKENIZER_NAME=$LLM_NAME
export METHOD="h2t-mapping"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../prompting.sbatch