#!/bin/bash
#SBATCH --job-name=train_speechllm
#SBATCH --partition=frida
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=H100:8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/speechllm/error_%x_%j.txt

EXPERIMENT_NAME=wer

NUM_GPUS=8
MICRO_BATCH_SIZE=8

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

# fixes non getting relu function
echo "Replacing utils_funcs.py.."
curl -o /opt/NeMo/nemo/collections/nlp/parts/utils_funcs.py https://raw.githubusercontent.com/aklemen/NeMo/relu-support/nemo/collections/nlp/parts/utils_funcs.py

# fixes non getting relu function
echo "Replacing mlp.py.."
curl -o /opt/NeMo/nemo/collections/nlp/modules/common/megatron/mlp.py https://raw.githubusercontent.com/aklemen/NeMo/relu-support/nemo/collections/nlp/modules/common/megatron/mlp.py


# train

# LLM_FILE_NAME=llama2-7b-bf16
# LLM_FILE_NAME=llama3_1_8b_instruct
# LLM_PATH="/models/llm/\${LLM_FILE_NAME}.nemo"

LLM_FILE_NAME=slopt_chat
LLM_PATH="/lpt-models/\${LLM_FILE_NAME}.nemo"

ASR_MODEL_FILE_NAME=conformer_ctc_bpe
ASR_MODEL_PATH="/models/asr/sl-SI_GEN_nemo-2.0/\${ASR_MODEL_FILE_NAME}.nemo"

ARTUR_MANIFESTS_PATH=/manifests/artur/kaj-je-transkript-tega-avdio-posnetka
TRAIN_MANIFEST="\${ARTUR_MANIFESTS_PATH}/clean/train.nemo"
VAL_MANIFEST="\${ARTUR_MANIFESTS_PATH}/clean/val.nemo"
VAL_NAME="artur-clean-val"

GLOBAL_BATCH_SIZE=\$(($NUM_GPUS * $MICRO_BATCH_SIZE))

NUM_TRAINING_SAMPLES=\$(wc -l < \${TRAIN_MANIFEST})
NUM_EPOCHS=2
STEPS_PER_EPOCH=\$((\${NUM_TRAINING_SAMPLES} / \${GLOBAL_BATCH_SIZE}))
ROUNDED_STEPS_PER_EPOCH=\$((((\${STEPS_PER_EPOCH} + 999) / 1000) * 1000))
MAX_STEPS=\$((\${ROUNDED_STEPS_PER_EPOCH} * \${NUM_EPOCHS}))

echo "================================================================"
echo "Micro batch size___$MICRO_BATCH_SIZE"
echo "Global batch size__\${GLOBAL_BATCH_SIZE}"
echo "Max steps__________\${MAX_STEPS}"
echo "================================================================"

MODEL_NAME=speechllm_\${ASR_MODEL_FILE_NAME}_\${LLM_FILE_NAME}_lora
MAX_DURATION=17 # for clean/train.nemo

cd "/opt/NeMo/examples/multimodal/speech_llm" || exit
python modular_audio_gpt_train.py --config-path="./conf" --config-name "modular_audio_gpt_config_peft" \
    name=\${MODEL_NAME} \
    trainer.devices=-1 \
    trainer.max_steps=\${MAX_STEPS} \
    trainer.val_check_interval=1000 \
    exp_manager.exp_dir="/exp" \
    model.freeze_audio_encoder=True \
    model.freeze_modality_adapter=False \
    model.freeze_llm=True \
    model.global_batch_size=\${GLOBAL_BATCH_SIZE} \
    model.micro_batch_size=$MICRO_BATCH_SIZE \
    model.pretrained_audio_model=\${ASR_MODEL_PATH} \
    model.restore_from_path=\${LLM_PATH} \
    model.data.train_ds.max_duration=\${MAX_DURATION} \
    ++model.data.train_ds.manifest_filepath=[\${TRAIN_MANIFEST}] \
    ++model.data.validation_ds.manifest_filepath=[\${VAL_MANIFEST}] \
    ++model.data.validation_ds.names=["\${VAL_NAME}"]
cd -

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT

LPT_DATASETS_PATH="/shared/workspace/lpt-llm/datasets"
LPT_MODELS_PATH="/shared/workspace/lpt-llm/models/lora_models"
EXP_PATH=${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/${version}

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:24.07"
CONTAINER_MOUNTS="$LPT_DATASETS_PATH:/dataset:ro,$LPT_MODELS_PATH:/lpt-models:ro,$EXP_PATH:/exp,${HOME}/manifests:/manifests:ro,${HOME}/models:/models:ro"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="${EXP_PATH}/${RUN_NAME}.%s.txt" \
  --container-workdir="/exp" \
  /exp/${RUN_NAME}.sh