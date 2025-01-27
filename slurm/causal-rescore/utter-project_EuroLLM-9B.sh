export LLM_NAME="utter-project/EuroLLM-9B"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.519 0.558 0.517 0.516"
export BETAS="0.114 0.136 0.301 0.162"
export BATCH_SIZE=64

sbatch --job-name=causal-rescore-eurollm-9b causal-rescore.sbatch



