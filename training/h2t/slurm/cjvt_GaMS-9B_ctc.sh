export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/ctc-h2t"

export PER_DEVICE_BATCH_SIZE=1
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=128
export LORA_ALPHA=64

export MODEL_NAME="gams-9b-ctc-h2t-$LORA_RANK-$LORA_ALPHA"

JOB_NAME="train-h2t-$MODEL_NAME"
sbatch --job-name="$JOB_NAME" finetune_h2t.sbatch