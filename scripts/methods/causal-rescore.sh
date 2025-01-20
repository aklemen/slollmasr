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


# LLM_NAME="gordicaleksa/SlovenianGPT"
# TOKENIZER_NAME="mistralai/Mistral-7B-v0.1"
# ALPHAS="0.797 0.702 0.822 0.826"
# BETAS="-0.037 0.134 0.415 0.520"

# LLM_NAME="meta-llama/Llama-3.1-8B"
# TOKENIZER_NAME=$LLM_NAME
# ALPHAS="0.262 0.238 0.327 0.238"
# BETAS="0.230 0.066 0.243 0.101"

# LLM_NAME="cjvt/GaMS-1B"
# TOKENIZER_NAME=$LLM_NAME
# ALPHAS="0.574 0.603 0.649 0.649"
# BETAS="0.096 0.057 0.616 0.652"

LLM_NAME="meta-llama/Llama-3.1-70B"
TOKENIZER_NAME=$LLM_NAME
ALPHAS="0.262 0.238 0.327 0.238" # same as Llama-3.1-8B
BETAS="0.230 0.066 0.243 0.101" # same as Llama-3.1-8B

 python /slollmasr/main.py \
     --method "causal-rescorer" \
     --llm_name $LLM_NAME \
     --tokenizer_name $TOKENIZER_NAME \
     --manifest_file_path $MANIFEST_PATH \
     --beams_file_paths $BEAM_FILE_PATHS \
     --beam_sizes $BEAM_SIZES \
     --alphas $ALPHAS \
     --betas $BETAS \
     --results_dir_path "/beams/rescored/$DATASET"