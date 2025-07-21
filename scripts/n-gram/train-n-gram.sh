#!/bin/bash

# DON'T FORGET TO RUN prepare-for-n-gram-training.sh (installs required packages)

DATASET_PATH="/dataset/artur/v1.0/nemo/train.nemo"
OUT_DIR="/testing/n-gram"
MODEL_PATH="/models/asr/conformer_ctc_bpe.nemo"
NGRAM_LENGTH=4

mkdir -p $OUT_DIR

start=$(date +%s)

python /opt/NeMo/scripts/asr_language_modeling/ngram_lm/train_kenlm.py \
    nemo_model_file="$MODEL_PATH" \
    train_paths="[$DATASET_PATH]" \
    kenlm_model_file="${OUT_DIR}/artur_train_kenlm_${NGRAM_LENGTH}_gram.binary" \
    ngram_length="$NGRAM_LENGTH" \
    kenlm_bin_path="usr/local/kenlm/build/bin" \
    preserve_arpa=True \
    save_nemo=True \
    normalize_unk_nemo=False

end=$(date +%s)

echo "KenLM training completed in $(($end-$start)) seconds, model available at $OUT_DIR"