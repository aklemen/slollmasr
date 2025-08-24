export LLM_NAME="cjvt/GaMS-9B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="" # TODO
export BETAS="" # TODO
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch