export LLM_NAME="cjvt/GaMS-9B"
export DATASET_NAME="aklemen/whisper-ctc-h2t"
export MODEL_NAME="GaMS-9B-whisper-CTC-H2T-LoRA"

export PER_DEVICE_BATCH_SIZE=2
export TARGET_EFFECTIVE_BATCH_SIZE=128

export LORA_RANK=64
export LORA_ALPHA=64

JOB_NAME="train-h2t-$MODEL_NAME"
sbatch --job-name="$JOB_NAME" finetune_h2t.sbatch
