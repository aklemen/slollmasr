export LLM_NAME="cjvt/GaMS-2B"
export PROMPT_FORMAT="gemma"
export PER_DEVICE_BATCH_SIZE=4
export TARGET_EFFECTIVE_BATCH_SIZE=64

export LORA_RANK=128
export LORA_ALPHA=128

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=L4:4 \
  --mem-per-gpu=32G \
  train_speechlm2.sbatch