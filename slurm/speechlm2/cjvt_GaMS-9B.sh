export LLM_NAME="cjvt/GaMS-9B"
export PER_DEVICE_BATCH_SIZE=8
export TARGET_EFFECTIVE_BATCH_SIZE=128

JOB_NAME=speechlm2-$(echo $LLM_NAME | sed 's/\//-/g' | sed 's/_/-/g')
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=H100:4 \
  --mem-per-gpu=128G \
  train_speechlm2_gemma.sbatch