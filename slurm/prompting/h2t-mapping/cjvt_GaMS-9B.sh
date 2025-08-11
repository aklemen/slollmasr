export LLM_NAME="cjvt/GaMS-9B"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=h2t-mapping-cjvt-GaMS-9B h2t-mapping.sbatch