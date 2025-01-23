export LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE="32"

sbatch prompt-error-correct.sbatch --job-name=prompt-error-correct-llama-31-8b-instruct