export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/whisper-ctc-h2t"
export PER_DEVICE_BATCH_SIZE=2
export LORA_RANK=8
export LORA_ALPHA=16

sbatch --job-name=lora-gams-9b-whisper-ctc fine-tune.sbatch