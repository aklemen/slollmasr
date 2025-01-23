export LLM_NAME="google/gemma-2-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE="128"

sbatch prompt-error-correct.sbatch --job-name=prompt-error-correct-gemma-2-27b-it