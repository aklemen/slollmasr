#!/bin/bash
#SBATCH --job-name=download_librispeech
#SBATCH --partition=frida
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=/shared/home/anton.klemen/logs/download_librispeech/output_%x_%j.out
#SBATCH --error=/shared/home/anton.klemen/logs/download_librispeech/error_%x_%j.out

DATASETS_PATH="$HOME/datasets"
DATA_ROOT="/datasets/librispeech"

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.02.01"
CONTAINER_MOUNTS="$DATASETS_PATH:/datasets"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="nemo-container" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output=/shared/home/anton.klemen/logs/download_librispeech/output_%x_%j.out \
  --error=/shared/home/anton.klemen/logs/download_librispeech/error_%x_%j.out \
  bash -c "python /opt/NeMo/scripts/dataset_processing/get_librispeech_data.py --data_root $DATA_ROOT --data_sets ALL --num_workers 64 --log"