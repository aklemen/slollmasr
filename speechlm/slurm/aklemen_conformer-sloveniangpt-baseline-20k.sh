export LLM_NAME="aklemen/conformer-sloveniangpt-baseline-20k"

JOB_NAME="salm-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch