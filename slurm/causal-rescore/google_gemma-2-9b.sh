export LLM_NAME="google/gemma-2-9b"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.297 0.297 0.297 0.297"
export BETAS="0.457 0.216 -0.07 0.213"
export BATCH_SIZE="64"

sbatch causal-rescore.sbatch


