export LLM_NAME="meta-llama/Llama-3.3-70B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=256

sbatch prompt-error-correct.sbatch