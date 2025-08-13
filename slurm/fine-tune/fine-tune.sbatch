#!/bin/bash
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

# Check if all required environment variables are set
required_vars=(LLM_NAME PER_DEVICE_BATCH_SIZE DATASET_NAME LORA_RANK LORA_ALPHA)
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set."
    exit 1
  fi
done

EXPERIMENT_NAME="$(echo "$LLM_NAME" | sed 's/\//_/g')__$(echo "$DATASET_NAME" | sed 's/\//_/g')"

ACCELERATE_CONFIG_FILE="/slollmasr/slurm/fine-tune/accelerate-ddp.yaml"
NUM_GPUS=4
EPOCHS=5

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Please set it before running the script."
  exit 1
fi

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH_SCRIPT_FILE_PATH=$(realpath "$0")
fi

# time of running script
DATETIME=$(TZ="Europe/Ljubljana" date "+%Y-%m-%d___%H-%M-%S_%3N")

# experiment dir
EXPERIMENT_DIR="lora-fine-tune/$EXPERIMENT_NAME/$DATETIME"
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

# set unbuffered python for realtime container logging
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

export WANDB_API_KEY=$WANDB_API_KEY

RESULTS_DIR=\"/exp/results\"

pip install --upgrade transformers accelerate
pip install trl

if grep -q \"distributed_type: DEEPSPEED\" $ACCELERATE_CONFIG_FILE; then
  pip install deepspeed
fi

accelerate launch --config_file=$ACCELERATE_CONFIG_FILE --num_processes=$NUM_GPUS /slollmasr/sft_fine_tune.py \
  --llm_name $LLM_NAME \
  --prompt_completion_dataset_name $DATASET_NAME \
  --output_dir_path \$RESULTS_DIR \
  --run_name $EXPERIMENT_NAME \
  --lora_r $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout 0.05 \
  --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
  --epochs $EPOCHS

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT_FILE_PATH

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.07"

EXPERIMENTS="$LOCAL_EXPERIMENT_DIR:/exp"

DATASETS="/shared/workspace/lpt-llm/datasets:/dataset:ro"
SLOLLMASR="$HOME/slollmasr:/slollmasr:ro"

TESTING="$HOME/testing:/testing"
BEAMS="$HOME/beams:/beams"
HF_HOME="$HOME/.cache/huggingface:/hf-home"

CONTAINER_MOUNTS="$EXPERIMENTS,$DATASETS,$SLOLLMASR,$TESTING,$BEAMS,$HF_HOME"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="$LOCAL_EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  --export="HF_HOME=/hf-home" \
  "/exp/$SCRIPT_FILE_NAME"