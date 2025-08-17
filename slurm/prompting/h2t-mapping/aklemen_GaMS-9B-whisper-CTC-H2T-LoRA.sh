export LLM_NAME="aklemen/GaMS-9B-whisper-CTC-H2T-LoRA"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=h2t-mapping-GaMS-9B-whisper-CTC-H2T-LoRA h2t-mapping.sbatch