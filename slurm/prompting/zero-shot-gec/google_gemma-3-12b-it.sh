export LLM_NAME="google/gemma-3-12b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=zero-shot-gec-gemma-3-12b-it zero-shot-gec.sbatch