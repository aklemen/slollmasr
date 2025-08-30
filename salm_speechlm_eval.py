import argparse
import datetime
from pathlib import Path

import pandas as pd
import torch
import wandb

from speechlm.salm_speechlm import SalmSpeechLM
from torch_datasets.manifest_dataset import ManifestDataset
from utils.logger import Logger
from utils.metrics_calculator import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--extra_eos_token_id', type=int, required=False, default=None)
    parser.add_argument('--manifest_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--results_dir_paths', nargs='+', type=str, required=True)
    parser.add_argument('--evaluation_dir_path', type=str, required=True)

    parser.add_argument('--batch_size', type=int, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(args)
    Logger.info("===================================")

    Logger.info("============ TORCH INFO ===========")
    Logger.info(f"Torch version: {torch.__version__}")
    Logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        Logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            Logger.info(f"  {i}. {torch.cuda.get_device_name(i)}")
    Logger.info("===================================")

    if len(args.manifest_file_paths) != len(args.results_dir_paths):
        raise ValueError(
            f"The number of manifest_file_paths ({len(args.manifest_file_paths)}) and results_dir_paths ({len(args.results_dir_paths)})"
            f"should be the same!"
        )

    Logger.info("Instantiating MetricsCalculator ...")
    calc = MetricsCalculator()

    Logger.info(f"Instantiating SalmSpeechLM' ...")
    speechlm = SalmSpeechLM(args.llm_name, args.batch_size, [args.extra_eos_token_id])

    eval_df = pd.DataFrame(columns=[
        'results_file',
        'wer',
        'cer',
        'run_duration',
        'run_duration_in_seconds',
        'rtfx',
        'gpus'
    ])

    Logger.info("Initializing WandB run ...")
    llm_base_name = args.llm_name.split('/')[-1]
    run_name = f"SpeechLM_{llm_base_name}"
    wandb_run = wandb.init(
        project="ASR+LLM",
        name=run_name,
        config=vars(args),
        tags=["speechlm", llm_base_name],
        group="speechlm"
    )


    for manifest_file_path, results_dir_path in zip(
            args.manifest_file_paths,
            args.results_dir_paths,
    ):
        Logger.info(f"Processing manifest file {manifest_file_path} ...")
        manifest = ManifestDataset(manifest_file_path)
        ground_truths = manifest.get_transcripts()

        new_best_hypotheses, run_duration = speechlm.run(manifest_file_path)

        results_df = pd.DataFrame({
            'ground_truth': ground_truths,
            'asr+llm_best_hypothesis': new_best_hypotheses,
        })

        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        results_file_name = f'{args.llm_name}'.replace('/', '_')
        results_file_path = f'{results_dir_path}/{results_file_name}.tsv'
        results_df.to_csv(results_file_path, sep='\t', index=False)
        Logger.info(f"Results saved to {results_file_path}!")

        wer, cer = calc.calculate_wer_and_cer(new_best_hypotheses, ground_truths)
        rtfx = calc.calculate_rtfx(manifest.get_dataset_duration_in_seconds(), run_duration)
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        new_eval_df = pd.DataFrame({
            'results_file': [results_file_path],
            'wer': [wer],
            'cer': [cer],
            'run_duration': [str(datetime.timedelta(seconds=run_duration))],
            'run_duration_in_seconds': [round(run_duration, 3)],
            'rtfx': [round(rtfx, 3)],
            'gpus': ', '.join(gpu_names)
        })
        if eval_df is None or eval_df.empty:
            eval_df = new_eval_df
        else:
            eval_df = pd.concat([eval_df, new_eval_df], ignore_index=True)
        eval_df.to_csv(f'{args.evaluation_dir_path}/evaluation.tsv', sep='\t', index=False)

        dataset_name = Path(manifest_file_path).stem
        wandb.log({
            "dataset": dataset_name,
            "wer": wer,
            "cer": cer,
            "runtime/seconds": round(run_duration, 3),
            "runtime/rtfx": round(rtfx, 3),
        })

        Logger.info(eval_df.to_string())

    wandb.log({
        "evaluation_table": wandb.Table(dataframe=eval_df)
    })
    wandb_run.finish()
