export LLM_NAME="meta-llama/Llama-3.1-70B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.404 0.389 0.388 0.404"
export BETAS="0.093 0.150 0.165 0.112"
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../../rescoring.sbatch