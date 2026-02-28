export LLM_NAME="aklemen/conformer-sloveniangpt-baseline"
export DO_SAMPLE=1

JOB_NAME="salm-nucleus-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch
