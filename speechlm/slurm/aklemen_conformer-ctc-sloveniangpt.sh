export LLM_NAME="aklemen/conformer-ctc-sloveniangpt"

JOB_NAME="speechlm-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch