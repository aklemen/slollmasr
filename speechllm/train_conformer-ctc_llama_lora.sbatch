#!/bin/bash
#SBATCH --job-name=train-speechllm-conformer-ctc-llama
#SBATCH --partition=frida
#SBATCH --time=??? #TODO
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --tasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out

# NeMo
NEMO_REPOSITORY_PATH="/NeMo"
NEMO_SPEECHLLM_EXAMPLES_PATH="$NEMO_REPOSITORY_PATH/examples/multimodal/speech_llm"

SLOLLMASR_REPOSITORY_PATH="/slollmasr"

# models
LLM_PATH="/models/llm/llama3_1_8b.nemo"
#LLM_PATH="/models/llm/llama3_1_8b_instruct.nemo"
ASR_MODEL_PATH="/models/asr/sl-SI_GEN_nemo-2.0/conformer_ctc_bpe.nemo"

# dataset
NEMO_ARTUR_PATH="/dataset/artur/v1.0/nemo"
TRAIN_MANIFESTS="[$NEMO_ARTUR_PATH/train.nemo]"
VAL_MANIFESTS="[$NEMO_ARTUR_PATH/dev.nemo]"
VAL_NAMES="[artur-dev]"

# config
CONFIGS_PATH="$SLOLLMASR_REPOSITORY_PATH/speechllm/configs"
CONFIG_NAME="modular_audio_gpt_config_peft"

CONTEXT_FILE_PATH="$SLOLLMASR_REPOSITORY_PATH/speechllm/context_list"

# training
# global_batch_size = micro_batch_size(=2) * num_gpus_per_node(=???) * num_nodes(=1) * accumulate_grad_batches(=1)
# micro_batch_size = batch_size_per_gpu
# TODO - set batch sizes, depending on GPUs usage
cd "$NEMO_SPEECHLLM_EXAMPLES_PATH" || exit
python modular_audio_gpt_train.py --config-path="$CONFIGS_PATH" --config-name "$CONFIG_NAME" \
    name="speechllm_conformer-ctc_llama31_8b_lora" \
    trainer.devices=-1 \
    model.freeze_audio_encoder=True \
    model.freeze_modality_adapter=False \
    model.freeze_llm=False \
    model.global_batch_size=4 \
    model.micro_batch_size=2 \
    model.pretrained_audio_model=$ASR_MODEL_PATH \
    model.restore_from_path=$LLM_PATH \
    model.data.train_ds.manifest_filepath="$TRAIN_MANIFESTS" \
    model.data.validation_ds.manifest_filepath="$VAL_MANIFESTS" \
    ++model.data.validation_ds.names="$VAL_NAMES" \
    ++model.data.train_ds.context_file="$CONTEXT_FILE_PATH" \
    ++model.data.train_ds.context_key="input_text" \
    ++model.data.train_ds.answer_key="text" \
    ++model.data.train_ds.prompt_template="'Q: {input_text}\nA: {text}'"