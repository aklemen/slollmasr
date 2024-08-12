#!/bin/bash
# TODO - adjust resources and calculate time
#SBATCH --job-name=train_speechllm_conformer-ctc_llama_lora
#SBATCH --partition=frida
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --tasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out

DATASETS_PATH="/shared/workspace/lpt-llm/datasets"
MANIFESTS_PATH="/shared/home/anton.klemen/manifests"
MODELS_PATH="/shared/home/anton.klemen/models"
SLOLLMASR_PATH="/shared/home/anton.klemen/slollmasr"
NEMO_PATH="/shared/home/anton.klemen/NeMo"

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:24.07"
CONTAINER_NAME="nemo-container"
CONTAINER_MOUNTS="$DATASETS_PATH:/dataset:ro,$MANIFESTS_PATH:/manifests:ro,$MODELS_PATH:/models:ro,$SLOLLMASR_PATH:/slollmasr:ro,$NEMO_PATH:/NeMo:ro"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$CONTAINER_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  bash /shared/home/anton.klemen/slollmasr/speechllm/train_conformer-ctc_llama_lora.sh