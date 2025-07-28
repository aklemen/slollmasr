#!/bin/bash
#SBATCH --job-name=infer_speechllm_checkpoints
#SBATCH --partition=frida
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100_80GB:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/speechllm/error_%x_%j.txt

EXPERIMENT_NAME=slovenian-gpt-infer-checkpoint

NUM_GPUS=1
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=8

LLM_FILE_NAME=slovenian_gpt
LLM_PATH="/models/llm/$LLM_FILE_NAME.nemo"

ASR_MODEL_FILE_NAME=conformer_ctc_bpe
ASR_MODEL_PATH="/models/asr/$ASR_MODEL_FILE_NAME.nemo"

TRAIN_EXP_DIR="$HOME/exp/slovenian-gpt-bf16-earlystop/2024-08-25___15-58/speechllm_conformer_ctc_bpe_slovenian_gpt_lora"
TRAIN_EXP_MOUNT="$TRAIN_EXP_DIR:/train-exp"
CHECKPOINT_PATH="/train-exp/checkpoints/speechllm_conformer_ctc_bpe_slovenian_gpt_lora--validation_loss_0.060-step_38590-epoch_5.ckpt"
HPARAMS_PATH="/train-exp/version_0/hparams.yaml"

MANIFESTS_DIR="/manifests/artur/kaj-je-transkript-tega-avdia/clean"
TEST_MANIFESTS=["/manifests/artur/kaj-je-transkript-tega-avdia/test.nemo"]
TEST_NAMES=["artur-test"]


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

python modular_audio_gpt_eval.py --config-path="./conf" --config-name "modular_audio_gpt_config_eval" \
    model.restore_from_path=$LLM_PATH \
    model.peft.restore_from_path="$CHECKPOINT_PATH" \
    model.peft.restore_from_hparams_path="$HPARAMS_PATH" \
    model.data.test_ds.names=$TEST_NAMES \
    model.data.test_ds.metric.name="wer" \
    model.data.test_ds.global_batch_size=$GLOBAL_BATCH_SIZE \
    model.data.test_ds.micro_batch_size=$MICRO_BATCH_SIZE \
    model.data.test_ds.tokens_to_generate=256 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.manifest_filepath=$TEST_MANIFESTS \
    ++model.data.test_ds.output_dir=/exp
    
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

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.07"
CONTAINER_MOUNTS="$EXP_MOUNT,$LPT_DATASETS_MOUNT,$MANIFESTS_MOUNT,$MODELS_MOUNT,$TESTING_MOUNT,$TRAIN_EXP_MOUNT"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="${EXP_PATH}/${RUN_NAME}.%s.txt" \
  --container-workdir="/exp" \
  /exp/${RUN_NAME}.sh