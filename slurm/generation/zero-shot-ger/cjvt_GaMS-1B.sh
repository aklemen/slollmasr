export LLM_NAME="cjvt/GaMS-1B-Chat"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=zero-shot-ger-cjvt-GaMS-1B-Chat zero-shot-ger.sbatch