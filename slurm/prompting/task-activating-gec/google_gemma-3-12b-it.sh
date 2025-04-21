export LLM_NAME="google/gemma-3-12b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=64

sbatch --job-name=task-activating-gec-gemma-3-12b-it task-activating-gec.sbatch