export LLM_NAME="meta-llama/Llama-3.1-8B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.262 0.238 0.327 0.238"
export BETAS="0.230 0.066 0.243 0.101"

sbatch --job-name=simple-causal-rescore-llama-31-8b simple-causal-rescore.sbatch