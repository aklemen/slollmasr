export LLM_NAME="aklemen/H2T-LoRA-test"
export TOKENIZER_NAME="cjvt/GaMS-9B"
export BATCH_SIZE=128

sbatch --job-name=h2t-mapping-cjvt-GaMS-9B-H2T h2t-mapping.sbatch