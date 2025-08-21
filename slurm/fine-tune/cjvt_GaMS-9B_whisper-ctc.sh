export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/whisper-ctc-h2t"
export MODEL_NAME="GaMS-9B-whisper-CTC-H2T-LoRA"

export PER_DEVICE_BATCH_SIZE=2
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=128
export LORA_ALPHA=64

JOB_NAME=h2t-$(echo $MODEL_NAME | sed 's/\//-/g' | sed 's/_/-/g')
sbatch --job-name="$JOB_NAME" fine-tune.sbatch