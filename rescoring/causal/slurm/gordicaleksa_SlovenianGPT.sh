export LLM_NAME="gordicaleksa/SlovenianGPT"
export TOKENIZER_NAME="mistralai/Mistral-7B-v0.1"
export ALPHAS="0.797 0.702 0.822 0.826"
export BETAS="-0.037 0.134 0.415 0.520"
export METHOD="causal-rescore"

JOB_NAME="$METHOD-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" ../../rescoring.sbatch