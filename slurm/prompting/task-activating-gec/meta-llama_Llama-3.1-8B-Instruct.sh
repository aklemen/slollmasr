export LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=64

sbatch --job-name=task-activating-gec-llama-31-8b-instruct task-activating-gec.sbatch