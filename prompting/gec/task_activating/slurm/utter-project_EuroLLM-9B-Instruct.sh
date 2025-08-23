export LLM_NAME="utter-project/EuroLLM-9B-Instruct"
export TOKENIZER_NAME=$LLM_NAME
export BATCH_SIZE=64

sbatch --job-name=task-activating-gec-eurollm-9b-instruct task-activating-gec.sbatch