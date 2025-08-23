python /slollmasr/training/h2t/build_prompt_completion_dataset_ctc.py \
  --manifest_file_path "/dataset/artur/v1.0/nemo/clean/train.nemo" \
  --beams_file_path "/exp/beam-search/2025-01-28___08-15-19_454/beams/artur/v1.0/nemo/clean/train/preds_out_width10_alpha1.0_beta0.0.tsv" \
  --output_dir_path "/testing/prompt-completion-dataset"