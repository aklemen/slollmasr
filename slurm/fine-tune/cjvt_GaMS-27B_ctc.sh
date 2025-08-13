export LLM_NAME="cjvt/GaMS-27B"
export DATASET_NAME="aklemen/ctc-h2t"
export PER_DEVICE_BATCH_SIZE=1
export LORA_RANK=128
export LORA_ALPHA=64
export MODEL_NAME="GaMS-27B-CTC-H2T-LoRA"

sbatch --job-name=lora-gams-27b-ctc fine-tune.sbatch