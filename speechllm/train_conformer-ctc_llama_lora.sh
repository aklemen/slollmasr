#!/bin/bash

# models
LLM_PATH="/models/llm/llama3_1_8b_instruct.nemo"
ASR_MODEL_PATH="/models/asr/sl-SI_GEN_nemo-2.0/conformer_ctc_bpe.nemo"

# manifests
TRAIN_MANIFESTS="[/manifests/artur/clean/train.nemo]"
VAL_MANIFESTS="[/manifests/artur/clean/val.nemo]"
VAL_NAMES="[artur-clean-val]"

# global_batch_size = micro_batch_size(=2) * num_gpus_per_node(=???) * num_nodes(=1) * accumulate_grad_batches(=1)
# micro_batch_size = batch_size_per_gpu
cd "/NeMo/examples/multimodal/speech_llm" || exit
# TODO - set batch sizes, depending on GPUs usage
python modular_audio_gpt_train.py --config-path="./conf" --config-name "modular_audio_gpt_config_peft" \
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
    ++model.data.train_ds.context_key="prompt" \
    ++model.data.train_ds.answer_key="text" \
    ++model.data.train_ds.prompt_template="'Q: {prompt}\nA: {text}'"