export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/ctc-h2t"
export MODEL_NAME="GaMS-9B-CTC-H2T-LoRA"

export PER_DEVICE_BATCH_SIZE=2
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=128
export LORA_ALPHA=64

sbatch --job-name=lora-gams-9b-ctc fine-tune.sbatch