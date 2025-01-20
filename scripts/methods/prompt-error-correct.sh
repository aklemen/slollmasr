#!/bin/bash

# DATASET="artur/v1.0/nemo/dev_without_utterance_AJO-3114-500056-0202"
# MANIFEST_PATH="/manifests/artur/dev_without_utterance_AJO-3114-500056-0202.nemo"

# DATASET="artur/v1.0/nemo/test"
# DATASET="commonvoice/v18.0/nemo/clean/all"
DATASET="fleurs/nemo/clean/all"
# DATASET="GVL/v4.2/nemo/all"
# DATASET="Sofes/v1.0/nemo/all"
# DATASET="voxpopuli/nemo/clean/all"
MANIFEST_PATH="/dataset/$DATASET.nemo"

BEAM_FILE_PATHS="/beams/original/$DATASET/preds_out_width5_alpha1.0_beta0.0.tsv /beams/original/$DATASET/preds_out_width10_alpha1.0_beta0.0.tsv /beams/original/$DATASET/preds_out_width50_alpha1.0_beta0.0.tsv /beams/original/$DATASET/preds_out_width100_alpha1.0_beta0.0.tsv"
BEAM_SIZES="5 10 50 100"


LLM_NAME="meta-llama/Llama-3.1-8B-Instruct"
# LLM_NAME="meta-llama/Llama-3.3-70B-Instruct"
# LLM_NAME="cjvt/GaMS-1B"

 python /slollmasr/main.py \
     --method "prompt-error-corrector" \
     --llm_name $LLM_NAME \
     --manifest_file_path $MANIFEST_PATH \
     --beams_file_paths $BEAM_FILE_PATHS \
     --beam_sizes $BEAM_SIZES \
     --results_dir_path "/beams/rescored/$DATASET"