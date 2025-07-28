#!/bin/bash

# DON'T FORGET TO RUN prepare_for_beam_search_with_n-gram.sh (installs required packages, fixes errors)

DATASET_PATH="artur/v1.0/nemo/dev"
# DATASET_PATH="artur/v1.0/nemo/test"
# DATASET_PATH="GVL/v4.2/nemo/all"
# DATASET_PATH="Sofes/v1.0/nemo/all"
# DATASET_PATH="commonvoice/v18.0/nemo/clean/all"
# DATASET_PATH="fleurs/nemo/clean/all"
# DATASET_PATH="voxpopuli/nemo/clean/all"

MANIFEST="/dataset/$DATASET_PATH.nemo"
OUT_DIR="/beams/n-gram/$DATASET_PATH"

MODEL_PATH="/models/asr/conformer_ctc_bpe.nemo"
KENLM_MODEL="/models/n-gram/artur_train_kenlm_4_gram.binary"

mkdir -p $OUT_DIR

start=$(date +%s)

python /opt/NeMo/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py \
    nemo_model_file="$MODEL_PATH" \
    kenlm_model_file="$KENLM_MODEL" \
    input_manifest="$MANIFEST" \
    preds_output_folder="$OUT_DIR" \
    hyps_cache_file="$OUT_DIR/cache_file" \
    decoding_mode="beamsearch_ngram" \
    decoding_strategy="beam" \
    beam_width=[5,10,50,100] \
    batch_size=256 \
    use_amp=True

end=$(date +%s)

echo "Beam search completed in $(($end-$start)) seconds, beams available at $OUT_DIR"