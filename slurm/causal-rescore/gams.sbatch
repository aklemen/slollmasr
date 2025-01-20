#!/bin/bash
#SBATCH --job-name=causal-rescore-gams
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100_80GB:2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/causal_rescore/error_%x_%j.txt

EXPERIMENT_NAME=causal-rescore-gams

LLM_NAME="cjvt/GaMS-1B"
TOKENIZER_NAME="cjvt/GaMS-1B"

BEAM_SIZES="5 10 50 100"
ALPHAS="0.576 0.595 0.65 0.586"
BETAS="0.080 0.092 0.49 0.201"

TEST_DATASETS=(
  "artur/v1.0/nemo/test"
  "commonvoice/v18.0/nemo/clean/all"
  "fleurs/nemo/clean/all"
  "GVL/v4.2/nemo/all"
  "Sofes/v1.0/nemo/all"
  "voxpopuli/nemo/clean/all"
)

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH_SCRIPT_FILE_PATH=$(realpath "$0")
fi

# time of running script
DATETIME=$(date -d "+1 hour" "+%Y-%m-%d___%H-%M")

# experiment dir
EXPERIMENT_DIR="${HOME}/exp/$EXPERIMENT_NAME/$DATETIME"
mkdir -p "$EXPERIMENT_DIR"

# archive this bash script
cp -rp "$SBATCH_SCRIPT_FILE_PATH" "$EXPERIMENT_DIR/script.sbatch"

# create execution script file
SCRIPT_FILE_PATH="$EXPERIMENT_DIR/script.sh"
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
\$(env | grep -i Slurm | sort | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\$(cat /etc/nccl.conf | sed \"s/^/# \${SLURMD_NODENAME}.\${SLURM_PROCID}>   /g\")
\"\"\"

# set unbuffered python for realtime container logging
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

for dataset in \"${TEST_DATASETS[@]}\"; do
  echo \"Processing dataset: '$dataset' ...\"
  MANIFEST_PATH=/dataset/$dataset.nemo
  BEAM_FILE_PATHS_ARRAY=()
  for size in \"${BEAM_SIZES[@]}\"; do
      BEAM_FILE_PATHS_ARRAY+=("/beams/original/$dataset/preds_out_width${size}_alpha1.0_beta0.0.tsv")
  done
  BEAM_FILE_PATHS=$(IFS=' '; echo "${BEAM_FILE_PATHS_ARRAY[*]}")

  python /slollmasr/main.py \
     --method causal-rescorer \
     --llm_name $LLM_NAME \
     --tokenizer_name $TOKENIZER_NAME \
     --manifest_file_path $MANIFEST_PATH \
     --beams_file_paths $BEAM_FILE_PATHS \
     --beam_sizes $BEAM_SIZES \
     --alphas $ALPHAS \
     --betas $BETAS \
     --results_dir_path /beams/rescored/$dataset
done

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT_FILE_PATH

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:24.12"

EXPERIMENTS="$HOME/exp:/exp"

DATASETS="/shared/workspace/lpt-llm/datasets:/dataset:ro"
MANIFESTS="$HOME/manifests:/manifests:ro"
MODELS="$HOME/models:/models:ro"
SLOLLMASR="$HOME/slollmasr:/slollmasr:ro"

BEAMS="$HOME/beams:/beams"
TESTING="$HOME/testing:/testing"
HF_HOME="$HOME/.cache/huggingface:/hf-home"

CONTAINER_MOUNTS="$EXPERIMENTS,$DATASETS,$MANIFESTS,$MODELS,$SLOLLMASR,$BEAMS,$TESTING,$HF_HOME"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="$EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  "$SCRIPT_FILE_PATH"