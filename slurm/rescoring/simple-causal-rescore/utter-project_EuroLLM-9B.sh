export LLM_NAME="utter-project/EuroLLM-9B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.519 0.558 0.517 0.516"
export BETAS="0.114 0.136 0.301 0.162"

sbatch --job-name=simple-causal-rescore-eurollm-9b simple-causal-rescore.sbatch