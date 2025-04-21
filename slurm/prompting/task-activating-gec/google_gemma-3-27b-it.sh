export LLM_NAME="google/gemma-3-27b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=32

sbatch --job-name=task-activating-gec-gemma-3-27b-it task-activating-gec.sbatch