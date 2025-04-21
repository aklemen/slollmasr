export LLM_NAME="google/gemma-3-27b-pt"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.479 0.499 0.497 0.511"
export BETAS="0.013 0.003 0.001 0.028"

sbatch --job-name=causal-rescore-gemma-3-27b-pt causal-rescore.sbatch