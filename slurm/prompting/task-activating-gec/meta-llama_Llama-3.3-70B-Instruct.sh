export LLM_NAME="meta-llama/Llama-3.3-70B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=32

sbatch --job-name=task-activating-gec-llama-33-70b-instruct task-activating-gec.sbatch