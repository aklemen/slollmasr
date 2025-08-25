export LLM_NAME="cjvt/GaMS-9B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.683 0.698 0.698 0.695"
export BETAS="0.235 0.286 0.286 0.481"
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch