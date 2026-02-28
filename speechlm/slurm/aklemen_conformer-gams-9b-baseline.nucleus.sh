export LLM_NAME="aklemen/conformer-gams-9b-baseline"
export EXTRA_EOS_TOKEN_ID="107"
export DO_SAMPLE=1

JOB_NAME="salm-nucleus-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch
