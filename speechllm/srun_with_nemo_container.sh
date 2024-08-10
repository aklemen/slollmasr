DATASETS_PATH="/shared/workspace/lpt-llm/datasets"
MODELS_PATH="/shared/home/anton.klemen/models"
SLOLLMASR_PATH="/shared/home/anton.klemen/slollmasr"
NEMO_PATH="/shared/home/anton.klemen/NeMo"

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:24.07"
CONTAINER_NAME="nemo-container"
CONTAINER_MOUNTS="$DATASETS_PATH:/dataset:ro,$MODELS_PATH:/models:ro,$SLOLLMASR_PATH:/slollmasr:ro,$NEMO_PATH:/NeMo:ro"

srun -p dev -w ana --gres=gpu \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$CONTAINER_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --pty bash