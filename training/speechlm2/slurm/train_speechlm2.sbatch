#!/bin/bash
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

# Check if all required environment variables are set
required_vars=(SLURM_GPUS_ON_NODE WANDB_API_KEY LLM_NAME PROMPT_FORMAT LORA_RANK LORA_ALPHA)
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set."
    exit 1
  fi
done

# Set defaults for optional variables
: "${RUN_NAME:=$(basename "$LLM_NAME")_r${LORA_RANK}-a${LORA_ALPHA}}"
: "${FREEZE_ENCODER:=false}"

EXPERIMENT_NAME="$RUN_NAME"

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH_SCRIPT_FILE_PATH=$(realpath "$0")
fi

# time of running script
DATETIME=$(TZ="Europe/Ljubljana" date "+%Y-%m-%d___%H-%M-%S_%3N")

# experiment dir
EXPERIMENT_DIR="salm-ablations/$EXPERIMENT_NAME/$DATETIME"
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

echo \"Training config:\"
echo \"  FREEZE_ENCODER=${FREEZE_ENCODER}\"
echo \"  LORA_RANK=${LORA_RANK}\"
echo \"  LORA_ALPHA=${LORA_ALPHA}\"

# set unbuffered python for realtime container logging
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

export WANDB_API_KEY=\"$WANDB_API_KEY\"

# Export env vars for OmegaConf interpolation
export LORA_RANK=\"$LORA_RANK\"
export LORA_ALPHA=\"$LORA_ALPHA\"

IS_RANK_0_DONE=/tmp/rank_0_done

if [[ \"\${SLURM_PROCID:-0}\" == \"0\" ]]; then
  echo \"# Rank 0 checking out 'main' branch to get newer SpeechLM2 code ...\"
  cd /opt/NeMo
  git branch --show-current
  git fetch origin main
  git checkout main
  git reset --hard ddcb75fb0237d0384f5cfbb50414da609662cb07
  git branch --show-current
  cd -
  echo \"# Rank 0 patching SpeechLM2 invalid device string issue ...\"
  curl -o /opt/NeMo/nemo/collections/speechlm2/parts/pretrained.py https://raw.githubusercontent.com/aklemen/NeMo/refs/heads/speechlm2-fixes/nemo/collections/speechlm2/parts/pretrained.py
  touch \"\${IS_RANK_0_DONE}\"
else
  echo \"# Rank \${SLURM_PROCID} waiting for NeMo install\"
  while [[ ! -f \"\${IS_RANK_0_DONE}\" ]]; do sleep 2; done
fi

# Build CLI overrides for list-type parameters
EXTRA_ARGS=\"\"

# If FREEZE_ENCODER is false, add prevent_freeze_params to unfreeze encoder
if [ \"${FREEZE_ENCODER}\" = \"false\" ]; then
  EXTRA_ARGS=\"\${EXTRA_ARGS} model.prevent_freeze_params='[\\\"^perception\\\\.preprocessor\\\\..+\$\\\", \\\"^perception\\\\.encoder\\\\..+\$\\\"]'\"
fi

python /opt/NeMo/examples/speechlm2/salm_train.py \\
  --config-path=/slollmasr/training/speechlm2 \\
  --config-name=salm.yaml \\
  model.pretrained_llm=${LLM_NAME} \\
  model.prompt_format=${PROMPT_FORMAT} \\
  exp_manager.name=${EXPERIMENT_NAME} \\
  exp_manager.wandb_logger_kwargs.name=${EXPERIMENT_NAME} \\
  \${EXTRA_ARGS}

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
