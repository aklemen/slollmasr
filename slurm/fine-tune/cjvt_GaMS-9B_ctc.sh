export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/ctc-h2t"
export PER_DEVICE_BATCH_SIZE=2
export LORA_RANK=128
export LORA_ALPHA=64
export MODEL_NAME="GaMS-9B-CTC-H2T-LoRA"

sbatch --job-name=lora-gams-9b-ctc fine-tune.sbatch