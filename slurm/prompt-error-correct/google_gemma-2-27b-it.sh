export LLM_NAME="google/gemma-2-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=prompt-error-correct-gemma-2-27b-it prompt-error-correct.sbatch