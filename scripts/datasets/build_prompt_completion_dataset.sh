python /slollmasr/scripts/datasets/build_prompt_completion_dataset.py \
  --manifest_file_path "/dataset/artur/v1.0/nemo/clean/train.nemo" \
  --ctc_beams_file_path "/exp/beam-search/2025-01-28___08-15-19_454/beams/artur/v1.0/nemo/clean/train/preds_out_width10_alpha1.0_beta0.0.tsv" \
  --whisper_manifest_file_path "/beams/whisper/artur/train/whisper_manifest_cleaned.nemo" \
  --whisper_beams_file_path "/beams/whisper/artur/train/whisper_beams_cleaned.tsv" \
  --output_dir_path "/testing/prompt-completion-dataset"