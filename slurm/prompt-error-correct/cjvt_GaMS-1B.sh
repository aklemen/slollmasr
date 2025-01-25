export LLM_NAME="cjvt/GaMS-1B-Chat"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=prompt-error-correct-cjvt-GaMS-1B-Chat prompt-error-correct.sbatch