# manifests
TEST_MANIFESTS="[/manifests/artur/test.nemo]"
TEST_NAMES="[artur-test]"

SPEECHLLM_MODEL_PATH="" # TODO - set path to the SpeechLLM

OUTPUT_DIR="" # TODO - set output dir

# inference
cd "/NeMo/examples/multimodal/speech_llm" || exit
# TODO - set batch sizes, depending on GPUs usage
python modular_audio_gpt_eval.py \
    model.restore_from_path="$SPEECHLLM_MODEL_PATH" \
    ++model.data.test_ds.manifest_filepath="$TEST_MANIFESTS" \
    model.data.test_ds.names="$TEST_NAMES" \
    model.data.test_ds.global_batch_size=8 \
    model.data.test_ds.micro_batch_size=8 \
    model.data.test_ds.tokens_to_generate=256 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.output_dir="$OUTPUT_DIR"