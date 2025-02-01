export LLM_NAME="cjvt/GaMS-1B"
export TOKENIZER_NAME="cjvt/GaMS-1B"
export ALPHAS="0.576 0.595 0.65 0.586"
export BETAS="0.080 0.092 0.49 0.201"

sbatch --job-name=simple-causal-rescore-gams-1b simple-causal-rescore.sbatch