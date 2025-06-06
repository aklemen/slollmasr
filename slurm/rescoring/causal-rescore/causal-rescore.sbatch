#!/bin/bash
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=H100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
# SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

# Check if all required environment variables are set
required_vars=(LLM_NAME TOKENIZER_NAME)
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set."
    exit 1
  fi
done

EXPERIMENT_NAME=$(echo "$LLM_NAME" | sed 's/\//_/g')
BATCH_SIZE=64

BEAM_SIZES="5 10 50 100"

if [ "$SHOULD_LINEAR_SEARCH_ARTUR_DEV" = true ]; then
  echo "Running linear search on Artur dev dataset."
  TEST_DATASETS=("artur/v1.0/nemo/dev_without_utterance_AJO-3114-500056-0202")
  TEST_MANIFESTS=("/manifests/artur/dev_without_utterance_AJO-3114-500056-0202.nemo")
else
  echo "Running causal rescoring on test datasets."
  if [ -z "$ALPHAS" ] || [ -z "$BETAS" ]; then
    echo "Error: ALPHAS and BETAS must be set when SHOULD_LINEAR_SEARCH_ARTUR_DEV=false."
    exit 1
  fi
  WEIGHTS_ARGS="--alphas $ALPHAS --betas $BETAS"
  TEST_DATASETS=(
    "artur/v1.0/nemo/test"
    "commonvoice/v18.0/nemo/clean/all"
    "fleurs/nemo/clean/all"
    "GVL/v4.2/nemo/all"
    "Sofes/v1.0/nemo/all"
    "voxpopuli/nemo/clean/all"
  )
  TEST_MANIFESTS=()
  for DATASET in "${TEST_DATASETS[@]}"; do
    TEST_MANIFESTS+=("/dataset/$DATASET.nemo")
  done
fi

BASE_CONTAINER_RESULTS_PATH="/exp/results"
RESULT_DIR_PATHS_ARRAY=()
BEAM_FILE_PATHS_ARRAY=()
for DATASET in "${TEST_DATASETS[@]}"; do
  RESULT_DIR_PATHS_ARRAY+=("$BASE_CONTAINER_RESULTS_PATH/$DATASET")
  for SIZE in $BEAM_SIZES; do
      BEAM_FILE_PATHS_ARRAY+=("/beams/original/${DATASET}/preds_out_width${SIZE}_alpha1.0_beta0.0.tsv")
  done
done



# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH_SCRIPT_FILE_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH_SCRIPT_FILE_PATH=$(realpath "$0")
fi

# time of running script
DATETIME=$(date -d "+1 hour" "+%Y-%m-%d___%H-%M-%S_%3N")

# experiment dir
if [ "$SHOULD_LINEAR_SEARCH_ARTUR_DEV" = true ]; then
  EXPERIMENT_DIR="causal-rescore/linear-search/$EXPERIMENT_NAME/$DATETIME"
else
  EXPERIMENT_DIR="causal-rescore/$EXPERIMENT_NAME/$DATETIME"
fi
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

# Fix for bug causing Gemma 2 to not work with multiple GPUs - https://github.com/huggingface/transformers/issues/34706
if [[ "$LLM_NAME" == \"google/gemma-2\"* ]]; then
  pip install transformers==4.43.4
fi

python /slollmasr/main.py \
   --method causal-rescore \
   --llm_name $LLM_NAME \
   --tokenizer_name $TOKENIZER_NAME \
   --manifest_file_paths ${TEST_MANIFESTS[*]} \
   --beams_file_paths ${BEAM_FILE_PATHS_ARRAY[*]} \
   --beam_sizes $BEAM_SIZES \
   --results_dir_paths ${RESULT_DIR_PATHS_ARRAY[*]} \
   --evaluation_dir_path $BASE_CONTAINER_RESULTS_PATH \
   --batch_size $BATCH_SIZE \
   $WEIGHTS_ARGS

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT_FILE_PATH

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.04"

EXPERIMENTS="$LOCAL_EXPERIMENT_DIR:/exp"

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
  --output="$LOCAL_EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  --export="HF_HOME=/hf-home" \
  "/exp/$SCRIPT_FILE_NAME"