export LLM_NAME="google/gemma-3-12b-pt"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.423 0.405 0.423 0.409"
export BETAS="0.104 0.158 0.229 0.221"
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch