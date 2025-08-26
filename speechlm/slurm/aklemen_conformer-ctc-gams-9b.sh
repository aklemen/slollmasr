export LLM_NAME="aklemen/conformer-ctc-gams-9b"
export EXTRA_EOS_TOKENS=("<end_of_turn>")

JOB_NAME="speechlm-$(basename "$LLM_NAME")"
sbatch --job-name="$JOB_NAME" speechlm.sbatch