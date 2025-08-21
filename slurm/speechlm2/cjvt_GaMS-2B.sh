export LLM_NAME="cjvt/GaMS-2B"
export PER_DEVICE_BATCH_SIZE=4
export TARGET_EFFECTIVE_BATCH_SIZE=128

JOB_NAME=$(echo $LLM_NAME | sed 's/\//-/g' | sed 's/_/-/g')
sbatch \
  --job-name="$JOB_NAME" \
  --ntasks-per-node=4 \
  --gpus-per-node=L4:4 \
  --mem-per-gpu=32G \
  train_speechlm2_gemma.sbatch