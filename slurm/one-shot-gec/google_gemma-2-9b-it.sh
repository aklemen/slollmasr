export LLM_NAME="google/gemma-2-9b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=one-shot-ger-gemma-2-9b-it one-shot-ger.sbatch