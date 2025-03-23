#!/bin/bash
#SBATCH --job-name=train_speechllm_slovenian_gpt
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=H100:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/speechllm/error_%x_%j.txt

EXPERIMENT_NAME=slovenian-gpt-bf16-earlystop

NUM_GPUS=8
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=64
VALIDATION_INTERVAL_IN_STEPS=3000
WARMUP_STEPS=2000

LLM_FILE_NAME=slovenian_gpt
LLM_PATH="/models/llm/$LLM_FILE_NAME.nemo"

ASR_MODEL_FILE_NAME=conformer_ctc_bpe
ASR_MODEL_PATH="/models/asr/$ASR_MODEL_FILE_NAME.nemo"

MANIFESTS_DIR="/manifests/artur/kaj-je-transkript-tega-avdia/clean"
TRAIN_MANIFESTS=["$MANIFESTS_DIR/train.nemo"]
VAL_MANIFESTS=["$MANIFESTS_DIR/val.nemo"]
VAL_NAMES=["clean-val"]

MODEL_NAME=speechllm_"$ASR_MODEL_FILE_NAME"_"$LLM_FILE_NAME"_lora

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH=$(realpath "$0")
fi

# convert the --key=value arguments to variables
for argument in "$@"
do
  if [[ $argument == *"="* ]]; then
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare ${v,,}="${value}"
   fi
  fi
done

# time of running script
DATETIME=`date -d "+2 hours" "+%Y-%m-%d___%H-%M"`
version=${version:-$DATETIME} # if version is not set, use DATETIME as default

# experiment dir
EXPERIMENT_DIR=${HOME}/exp
mkdir -p ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${version}

# set run name
if [ "${version}" == "${DATETIME}" ]; then
  RUN_NAME=${version}
else
  RUN_NAME=${version}_R${DATETIME}
fi

# archive this script
cp -rp ${SBATCH} ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${version}/${RUN_NAME}.sbatch

# prepare execution script name
SCRIPT=${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${version}/${RUN_NAME}.sh
touch $SCRIPT
chmod a+x $SCRIPT

IS_DISTRIBUTED=$([ 1 -lt $SLURM_JOB_NUM_NODES ] && echo " distributed over $SLURM_JOB_NUM_NODES nodes" || echo " on 1 node")

# prepare the execution script content
echo """#!/bin/bash

# using `basename $SBATCH` -> $RUN_NAME.sbatch, running $SLURM_NPROCS tasks$IS_DISTRIBUTED

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


# https://github.com/NVIDIA/NeMo/pull/10115
echo "Replacing data_utils.py.."
curl -o /opt/NeMo/nemo/collections/multimodal/speech_llm/parts/utils/data_utils.py https://raw.githubusercontent.com/aklemen/NeMo/text-processing/nemo/collections/multimodal/speech_llm/parts/utils/data_utils.py


# train

cd "/opt/NeMo/examples/multimodal/speech_llm" || exit

python modular_audio_gpt_train.py --config-path="./conf" --config-name "modular_audio_gpt_config_peft" \
    name=$MODEL_NAME \
    trainer.devices=-1 \
    trainer.precision=bf16 \
    trainer.val_check_interval=$VALIDATION_INTERVAL_IN_STEPS \
    exp_manager.exp_dir="/exp" \
    exp_manager.create_early_stopping_callback=True \
    model.freeze_audio_encoder=True \
    model.freeze_llm=True \
    model.global_batch_size=$GLOBAL_BATCH_SIZE \
    model.micro_batch_size=$MICRO_BATCH_SIZE \
    model.pretrained_audio_model=$ASR_MODEL_PATH \
    model.restore_from_path=$LLM_PATH \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.validation_ds.names=$VAL_NAMES \
    model.optim.sched.warmup_steps=$WARMUP_STEPS
    
cd -

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT

EXP_PATH="$EXPERIMENT_DIR/$EXPERIMENT_NAME/$version"

EXP_MOUNT="$EXP_PATH:/exp"
LPT_DATASETS_MOUNT="/shared/workspace/lpt-llm/datasets:/dataset:ro"
MANIFESTS_MOUNT="$HOME/manifests:/manifests:ro"
MODELS_MOUNT="$HOME/models:/models:ro"
DATASETS_MOUNT="$HOME/datasets:/datasets:ro"
SLOLLMASR_MOUNT="$HOME/slollmasr:/slollmasr:ro"

BEAMS_MOUNT="$HOME/beams:/beams"
TESTING_MOUNT="$HOME/testing:/testing"
SCRIPTS_MOUNT="$HOME/scripts:/scripts"
HF_HOME_MOUNT="$HOME/.cache/huggingface:/hf-home"

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.02"
CONTAINER_MOUNTS="$EXP_MOUNT,$LPT_DATASETS_MOUNT,$MANIFESTS_MOUNT,$MODELS_MOUNT,$TESTING_MOUNT"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="${EXP_PATH}/${RUN_NAME}.%s.txt" \
  --container-workdir="/exp" \
  /exp/${RUN_NAME}.sh