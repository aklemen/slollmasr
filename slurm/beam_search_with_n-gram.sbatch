#!/bin/bash
#SBATCH --job-name=beam-search-n-gram
#SBATCH --partition=frida
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/dev/null
#SBATCH --error=/shared/home/anton.klemen/logs/error/%x/%j.log

EXPERIMENT_NAME="beam-search-4-gram"

BATCH_SIZE=512

TEST_DATASETS=(
#  "artur/v1.0/nemo/clean/train"
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
DATETIME=$(date -d "+1 hour" "+%Y-%m-%d___%H-%M-%S_%3N")

# experiment dir
EXPERIMENT_DIR="$EXPERIMENT_NAME/$DATETIME"
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

########################### PREPARE FOR BEAM SEARCH WITH N-GRAM LANGUAGE MODELING ###########################

## Install decoders

apt-get update && apt-get upgrade -y && apt-get install -y swig && rm -rf /var/lib/apt/lists/*

git clone https://github.com/NVIDIA/OpenSeq2Seq && \
cd OpenSeq2Seq && \
git checkout ctc-decoders && \
cd .. && \
mv OpenSeq2Seq/decoders /opt/NeMo/ && \
rm -rf OpenSeq2Seq && \
cd /opt/NeMo/decoders && \
cp /opt/NeMo/scripts/installers/setup_os2s_decoders.py ./setup.py && \
./setup.sh


## Install KenLM

apt-get install -y libeigen3-dev libboost-all-dev
cd /usr/local || exit
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j

pip3 install git+https://github.com/kpu/kenlm.git


# Replace eval_beamsearch_ngram_ctc.py with the one with time logging
curl -o /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py https://raw.githubusercontent.com/aklemen/NeMo/time-logging/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py


############################### RUN BEAM SEARCH WITH N-GRAM LANGUAGE MODELING ###############################

MODEL_PATH=/models/asr/conformer_ctc_bpe.nemo
TEST_DATASETS_ARRAY=(${TEST_DATASETS[*]})
KENLM_MODEL_PATH=/models/n-gram/artur_train_kenlm_4_gram.binary

for DATASET in \"\${TEST_DATASETS_ARRAY[@]}\"; do
  MANIFEST=\"/dataset/\$DATASET.nemo\"
  RESULT_DIR=\"/exp/beams/\$DATASET\"
  CACHE_FILE_PATH=\"/exp/beams/\$DATASET/probs_cache\"

  mkdir -p \$RESULT_DIR

  python /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py \
      nemo_model_file=\"\$MODEL_PATH\" \
      kenlm_model_file=\"\$KENLM_MODEL_PATH\" \
      input_manifest=\"\$MANIFEST\" \
      preds_output_folder=\"\$RESULT_DIR\" \
      hyps_cache_file=\"\$CACHE_FILE_PATH\" \
      decoding_mode=beamsearch_ngram \
      decoding_strategy=beam \
      beam_width=[5,10,50,100] \
      batch_size=$BATCH_SIZE \
      use_amp=True

  echo \"Finished beam search for \$DATASET. Beams saved to \$RESULT_DIR.\"
done

echo -e \"\"\"
# finished at \$(date)
\"\"\"
""" >> $SCRIPT_FILE_PATH

CONTAINER_IMAGE="nvcr.io/nvidia/nemo:25.07"

EXPERIMENTS="$LOCAL_EXPERIMENT_DIR:/exp"

DATASETS="/shared/workspace/lpt-llm/datasets:/dataset:ro"
MANIFESTS="$HOME/manifests:/manifests:ro"
MODELS="$HOME/models:/models:ro"

CONTAINER_MOUNTS="$EXPERIMENTS,$DATASETS,$MANIFESTS,$MODELS"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$SLURM_JOB_NAME" \
  --container-mounts="$CONTAINER_MOUNTS" \
  --output="$LOCAL_EXPERIMENT_DIR/output.log" \
  --container-workdir="/exp" \
  --export="HF_HOME=/hf-home" \
  "/exp/$SCRIPT_FILE_NAME"