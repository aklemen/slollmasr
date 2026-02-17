export LLM_NAME="aklemen/SlovenianGPT"
export PROMPT_FORMAT="mistral"
export PER_DEVICE_BATCH_SIZE=8
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=128
export LORA_ALPHA=128

JOB_NAME="train-speechlm2-$(basename "$LLM_NAME")"
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2.sbatch