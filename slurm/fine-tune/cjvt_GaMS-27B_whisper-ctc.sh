export LLM_NAME="cjvt/GaMS-27B"
export DATASET_NAME="aklemen/whisper-ctc-h2t"
export MODEL_NAME="GaMS-27B-whisper-CTC-H2T-LoRA"

export PER_DEVICE_BATCH_SIZE=1
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=128
export LORA_ALPHA=64

sbatch --job-name=lora-gams-27b-whisper-ctc fine-tune.sbatch