export LLM_NAME="google/gemma-2-9b"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.311 0.298 0.295 0.294"
export BETAS="0.463 0.222 -0.072 0.254"

sbatch --job-name=simple-causal-rescore-gemma-2-9b simple-causal-rescore.sbatch