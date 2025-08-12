export LLM_NAME="aklemen/H2T-LoRA"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=h2t-mapping-cjvt-GaMS-9B-H2T h2t-mapping.sbatch