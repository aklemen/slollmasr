WANDB_API_KEY="<your_wandb_api_key>"

pip install trl

python /slollmasr/fine_tune.py \
--llm_name "google/gemma-3-1b-pt" \
--manifest_file_path "/beams/whisper/artur/train/transcribed_manifest.nemo" \
--beams_file_path "/beams/whisper/artur/train/beams_10.tsv" \
--beam_size 10 \
--output_dir_path "/testing/fine-tune" \
--tokenized_dataset_dir_path "/testing/tokenized-artur-train-10" \
--run_name "test-run"