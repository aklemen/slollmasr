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
OUT_DIR="/beams/n-gram/gpu/$DATASET_PATH"

MODEL_PATH="/models/asr/conformer_ctc_bpe.nemo"
KENLM_MODEL="/models/n-gram/artur_train_kenlm_4_gram.binary.nemo"

mkdir -p $OUT_DIR

start=$(date +%s)

python /opt/NeMo/examples/asr/speech_to_text_eval.py \
    model_path="$MODEL_PATH" \
    amp=False \
    amp_dtype=bfloat16 \
    matmul_precision=high \
    compute_dtype=bfloat16 \
    presort_manifest=false \
    cuda=0 \
    batch_size=256 \
    dataset_manifest="$MANIFEST" \
    ctc_decoding.beam.ngram_lm_model="$KENLM_MODEL" \
    ctc_decoding.beam.ngram_lm_alpha=1 \
    ctc_decoding.beam.beam_size=5 \
    ctc_decoding.beam.beam_beta=0 \
    ctc_decoding.strategy="beam_batch" \
    ctc_decoding.beam.allow_cuda_graphs=True \
    output_filename="$OUT_DIR/evaluation_transcripts.json"

end=$(date +%s)

echo "Beam search completed in $(($end-$start)) seconds, beams available at $OUT_DIR"