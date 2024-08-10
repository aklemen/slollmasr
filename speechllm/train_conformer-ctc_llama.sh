# NeMo
NEMO_REPOSITORY_PATH="/NeMo"
NEMO_SPEECHLLM_EXAMPLES_PATH="$NEMO_REPOSITORY_PATH/examples/multimodal/speech_llm"

# slollmasr
SLOLLMASR_REPOSITORY_PATH="/slollmasr"

# models
LLM_PATH="/models/llm/llama3_1_8b.nemo"
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
TRAINING_SCRIPT="modular_audio_gpt_train.py"
cd "$NEMO_SPEECHLLM_EXAMPLES_PATH" || exit
python "$TRAINING_SCRIPT" --config-path="$CONFIGS_PATH" --config-name "$CONFIG_NAME" \
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
    model.data.validation_ds.names="$VAL_NAMES" \
    model.data.train_ds.context_file"$CONTEXT_FILE_PATH"