export LLM_NAME="google/gemma-3-12b-it"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=128

sbatch --job-name=one-shot-gec-gemma-3-12b-it one-shot-gec.sbatch