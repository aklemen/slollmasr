export LLM_NAME="google/gemma-2-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=64

sbatch --job-name=task-activating-gec-gemma-2-27b-it task-activating-gec.sbatch