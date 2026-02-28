export LLM_NAME="aklemen/conformer-gams-9b-baseline-20k"
export EXTRA_EOS_TOKEN_ID="107"

JOB_NAME="salm-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch