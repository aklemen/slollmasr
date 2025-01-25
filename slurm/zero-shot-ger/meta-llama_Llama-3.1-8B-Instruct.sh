export LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=zero-shot-ger-llama-31-8b-instruct zero-shot-ger.sbatch