export LLM_NAME="cjvt/GaMS-1B"
export TOKENIZER_NAME="cjvt/GaMS-1B"
export ALPHAS="0.576 0.595 0.65 0.586"
export BETAS="0.080 0.092 0.49 0.201"
export BATCH_SIZE="128"

sbatch --job-name=causal-rescore-gams-1b causal-rescore.sbatch