export LLM_NAME="EMBEDDIA/sloberta"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="" # TODO
export BETAS="" # TODO

sbatch --job-name=masked-rescore-sloberta ../masked-rescore.sbatch