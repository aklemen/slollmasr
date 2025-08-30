export LLM_NAME="google/gemma-3-27b-pt"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.481 0.454 0.492 0.495"
export BETAS="-0.005 -0.001 0.16 0.16"
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch