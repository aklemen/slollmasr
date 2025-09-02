export LLM_NAME="aklemen/conformer-ctc-sloveniangpt"
export EXTRA_EOS_TOKEN_ID="107"

JOB_NAME="speechlm-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch