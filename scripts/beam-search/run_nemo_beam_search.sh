# DON'T FORGET TO RUN prepare_for_beam_search.sh (installs required packages, fixes errors)

DATASET_PATH="artur/v1.0/nemo/dev"
# DATASET_PATH="artur/v1.0/nemo/all"
# DATASET_PATH="GVL/v4.2/nemo/all"
# DATASET_PATH="Sofes/v1.0/nemo/all"
# DATASET_PATH="commonvoice/v18.0/nemo/clean/all"
# DATASET_PATH="fleurs/nemo/clean/all"
# DATASET_PATH="voxpopuli/nemo/clean/all"

MANIFEST="/dataset/$DATASET_PATH.nemo"
OUT_DIR="/beams/original/$DATASET_PATH"

MODEL_PATH="/models/asr/conformer_ctc_bpe.nemo"

mkdir -p $OUT_DIR

start=$(date +%s)

python /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py \
    nemo_model_file="$MODEL_PATH" \
    input_manifest="$MANIFEST" \
    preds_output_folder="$OUT_DIR" \
    probs_cache_file="$OUT_DIR/cache_file" \
    decoding_mode=beamsearch \
    decoding_strategy="beam" \
    beam_width=[5,10,50,100] \
    beam_batch_size=256 \
    use_amp=True

end=$(date +%s)

echo "Beam search completed in $(($end-$start)) seconds, beams available at $OUT_DIR"