export LLM_NAME="/testing/h2t-test-model/adapter"
export TOKENIZER_NAME="/testing/h2t-test-model/tokenizer"
export BATCH_SIZE=128

sbatch --job-name=h2t-mapping-cjvt-GaMS-1B-H2T h2t-mapping.sbatch