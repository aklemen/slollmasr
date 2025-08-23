export LLM_NAME="EMBEDDIA/sloberta"
export TOKENIZER_NAME=$LLM_NAME
export ALPHAS="0.095 0.095 0.077 0.159"
export BETAS="-0.007 -0.159 -0.175 -0.359"
export METHOD="masked-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch