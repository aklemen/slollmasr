#!/bin/bash
#SBATCH --partition=frida
#SBATCH --job-name=oomptimizer
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=H100:1
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

set -euo pipefail

DATETIME=$(TZ="Europe/Ljubljana" date "+%Y-%m-%d___%H-%M-%S_%3N")
LOCAL_EXPERIMENT_DIR="$HOME/exp/oomptimizer/$DATETIME"
mkdir -p "$LOCAL_EXPERIMENT_DIR"

SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
cp -rp "$SBATCH_SCRIPT_FILE_PATH" "$LOCAL_EXPERIMENT_DIR/script.sbatch"

cp "$HOME/slollmasr/training/speechlm2/salm_slovenian_gpt.yaml" "$LOCAL_EXPERIMENT_DIR/"
cp "$HOME/slollmasr/training/speechlm2/salm_gams_9b.yaml" "$LOCAL_EXPERIMENT_DIR/"

SCRIPT_FILE_NAME="script.sh"
SCRIPT_FILE_PATH="$LOCAL_EXPERIMENT_DIR/$SCRIPT_FILE_NAME"

cat > "$SCRIPT_FILE_PATH" << 'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

echo "# Starting oomptimizer at $(date)"
echo "# Running on $(hostname)"
nvidia-smi

echo "# Checking out NeMo main branch..."
cd /opt/NeMo
git fetch origin main
git checkout main
git pull
cd -

export PYTHONFAULTHANDLER=1
export HF_HOME=/hf-home

OOMPTIMIZER_WRAPPER="/slollmasr/training/speechlm2/oomptimizer_wrapper.py"

echo ""
echo "========================================"
echo "Running oomptimizer for SlovenianGPT"
echo "========================================"
python "$OOMPTIMIZER_WRAPPER" \
  --module-name nemo.collections.speechlm2.SALM \
  --config-path /slollmasr/training/speechlm2/salm_slovenian_gpt.yaml \
  --buckets "[47,55,64,73,83,94,108,125,148]"

echo ""
echo "========================================"
echo "Running oomptimizer for GaMS-9B"
echo "========================================"
python "$OOMPTIMIZER_WRAPPER" \
  --module-name nemo.collections.speechlm2.SALM \
  --config-path /slollmasr/training/speechlm2/salm_gams_9b.yaml \
  --buckets "[44,52,59,67,76,87,99,114,136]"

echo ""
echo "# Finished at $(date)"
SCRIPT_EOF

chmod +x "$SCRIPT_FILE_PATH"

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.11"

EXPERIMENTS="$LOCAL_EXPERIMENT_DIR:/exp"
DATASETS="/shared/workspace/lpt-llm/datasets:/dataset:ro"
CUSTOM_DATASETS="$HOME/custom-datasets:/custom-datasets:ro"
MANIFESTS="$HOME/manifests:/manifests:ro"
MODELS="$HOME/models:/models:ro"
SLOLLMASR="$HOME/slollmasr:/slollmasr:ro"
HF_HOME="$HOME/.cache/huggingface:/hf-home"

CONTAINER_MOUNTS="$EXPERIMENTS,$DATASETS,$CUSTOM_DATASETS,$MANIFESTS,$MODELS,$SLOLLMASR,$HF_HOME"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="$LOCAL_EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  --export="HF_HOME=/hf-home" \
  "/exp/$SCRIPT_FILE_NAME"

echo "Output saved to: $LOCAL_EXPERIMENT_DIR/output.log"
