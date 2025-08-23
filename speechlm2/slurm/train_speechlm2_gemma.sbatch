#!/bin/bash
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

# Check if all required environment variables are set
required_vars=(SLURM_GPUS_ON_NODE WANDB_API_KEY LLM_NAME PER_DEVICE_BATCH_SIZE TARGET_EFFECTIVE_BATCH_SIZE)
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set."
    exit 1
  fi
done

GRADIENT_ACCUMULATION_STEPS=$((TARGET_EFFECTIVE_BATCH_SIZE / (SLURM_GPUS_ON_NODE * PER_DEVICE_BATCH_SIZE)))

EXPERIMENT_NAME=$(echo "$LLM_NAME" | sed 's/\//_/g')

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH_SCRIPT_FILE_PATH=$(realpath "$0")
fi

# time of running script
DATETIME=$(TZ="Europe/Ljubljana" date "+%Y-%m-%d___%H-%M-%S_%3N")

# experiment dir
EXPERIMENT_DIR="speechlm2/$EXPERIMENT_NAME/$DATETIME"
LOCAL_EXPERIMENT_DIR="$HOME/exp/$EXPERIMENT_DIR"
mkdir -p "$LOCAL_EXPERIMENT_DIR"

# archive this bash script
cp -rp "$SBATCH_SCRIPT_FILE_PATH" "$LOCAL_EXPERIMENT_DIR/script.sbatch"

# create execution script file
SCRIPT_FILE_NAME="script.sh"
SCRIPT_FILE_PATH="$LOCAL_EXPERIMENT_DIR/$SCRIPT_FILE_NAME"
touch "$SCRIPT_FILE_PATH"
chmod a+x "$SCRIPT_FILE_PATH"


# prepare the execution script content
echo """#!/bin/bash

# running $SLURM_NPROCS tasks

# prepare sub-script for debug outputs
echo -e \"\"\"
# starting at \$(date)
# running process \$SLURM_PROCID on \$SLURMD_NODENAME
\$(nvidia-smi | grep Version | sed -e 's/ *| *//g' -e \"s/   */\n# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\" -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(nvidia-smi -L | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import torch; print(f\"torch: {torch.__version__}\")' | sed -e \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import torch, torch.utils.collect_env; torch.utils.collect_env.main()' | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(python -c 'import transformers, accelerate; print(f\"transformers: {transformers.__version__}, accelerate: {accelerate.__version__}\")')
\$(env | grep -i Slurm | sort | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(cat /etc/nccl.conf | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\"\"\"

echo \"PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}\"
echo \"GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}\"
echo \"TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE}\"

# set unbuffered python for realtime container logging
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

export WANDB_API_KEY="$WANDB_API_KEY"

IS_RANK_0_DONE=/tmp/rank_0_done

if [[ \"\${SLURM_PROCID:-0}\" == \"0\" ]]; then
  echo \"# Rank 0 checking out 'main' branch to use most up-to-date SpeechLM2 code ...\"
  cd /opt/NeMo
  git branch --show-current
  git fetch origin main
  git reset --hard origin/main
  git branch --show-current
  cd -
  echo \"# Rank 0 patching SpeechLM2 invalid device string issue ...\"
  curl -o /opt/NeMo/nemo/collections/speechlm2/parts/pretrained.py https://raw.githubusercontent.com/aklemen/NeMo/refs/heads/speechlm2-fixes/nemo/collections/speechlm2/parts/pretrained.py
  touch \"\${IS_RANK_0_DONE}\"
else
  echo \"# Rank \${SLURM_PROCID} waiting for NeMo install\"
  while [[ ! -f \"\${IS_RANK_0_DONE}\" ]]; do sleep 2; done
fi

python /opt/NeMo/examples/speechlm2/salm_train.py \
  --config-path=/slollmasr/speechlm2 \
  --config-name=gemma_salm.yaml \
  model.pretrained_llm=${LLM_NAME} \
  trainer.accumulate_grad_batches=${GRADIENT_ACCUMULATION_STEPS} \
  data.train_ds.batch_size=${PER_DEVICE_BATCH_SIZE} \
  exp_manager.name=${EXPERIMENT_NAME} \
  exp_manager.wandb_logger_kwargs.name=${EXPERIMENT_NAME}

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT_FILE_PATH

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.07"

EXPERIMENTS="$LOCAL_EXPERIMENT_DIR:/exp"

DATASETS="/shared/workspace/lpt-llm/datasets:/dataset:ro"
CUSTOM_DATASETS="$HOME/custom-datasets:/custom-datasets:ro"
MANIFESTS="$HOME/manifests:/manifests:ro"
MODELS="$HOME/models:/models:ro"
SLOLLMASR="$HOME/slollmasr:/slollmasr:ro"

BEAMS="$HOME/beams:/beams"
TESTING="$HOME/testing:/testing"
HF_HOME="$HOME/.cache/huggingface:/hf-home"

CONTAINER_MOUNTS="$EXPERIMENTS,$DATASETS,$CUSTOM_DATASETS,$MANIFESTS,$MODELS,$SLOLLMASR,$BEAMS,$TESTING,$HF_HOME"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="$LOCAL_EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  --export="HF_HOME=/hf-home" \
  "/exp/$SCRIPT_FILE_NAME"