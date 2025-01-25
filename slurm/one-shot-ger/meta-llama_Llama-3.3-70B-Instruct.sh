export LLM_NAME="meta-llama/Llama-3.3-70B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=one-shot-ger-llama-33-70b-instruct one-shot-ger.sbatch