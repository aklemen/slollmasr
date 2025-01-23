export LLM_NAME="google/gemma-2-27b"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.386 0.388 0.446 0.462"
export BETAS="−0.029 −0.001 −0.005 0.001"
export BATCH_SIZE="32"

sbatch causal-rescore.sbatch