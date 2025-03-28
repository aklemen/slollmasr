import argparse
import datetime
import time
from pathlib import Path

import pandas as pd

from best_hypotheses_selector import BestHypothesesSelector
from logger import Logger
from methods.causal_rescorer import CausalReScorer
from methods.one_shot_gec import OneShotGec
from methods.simple_causal_rescorer import SimpleCausalReScorer
from methods.task_activating_gec import TaskActivatingGec
from methods.zero_shot_gec import ZeroShotGec
from methods.zero_shot_selection import ZeroShotSelection
from metrics_calculator import MetricsCalculator
from torch_datasets.hypotheses_dataset import HypothesesDataset
from torch_datasets.manifest_dataset import ManifestDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--beams_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--beam_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--results_dir_paths', nargs='+', type=str, required=True)
    parser.add_argument('--evaluation_dir_path', type=str, required=True)

    parser.add_argument('--alphas', nargs='+', type=float, required=False)
    parser.add_argument('--betas', nargs='+', type=float, required=False)

    parser.add_argument('--batch_size', type=int, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(args)
    Logger.info("===================================")

    if args.tokenizer_name is None:
        Logger.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    if args.alphas is None:
        args.alphas = [None] * len(args.beam_sizes)
    if args.betas is None:
        args.betas = [None] * len(args.beam_sizes)

    if len(args.beam_sizes) != len(args.alphas) or len(args.beam_sizes) != len(args.betas):
        raise ValueError(
            f"The number of beam_sizes ({len(args.beam_sizes)}), "
            f"alphas ({len(args.alphas)}) "
            f"and betas ({len(args.betas)}) should be the same!"
        )

    if len(args.beams_file_paths) < len(args.beam_sizes) or len(args.beams_file_paths) % len(args.beam_sizes) != 0:
        raise ValueError(
            f"The number of beams_file_paths ({len(args.beams_file_paths)}) should be "
            f"a multiple of the number of beam_sizes ({len(args.beam_sizes)})!"
        )

    num_of_datasets = len(args.beams_file_paths) // len(args.beam_sizes)
    if len(args.manifest_file_paths) != num_of_datasets or len(args.results_dir_paths) != num_of_datasets:
        raise ValueError(
            f"The number of manifest_file_paths ({len(args.manifest_file_paths)}) and results_dir_paths ({len(args.results_dir_paths)})"
            f"should be the same as len(args.beams_file_paths) // len(args.beam_sizes) ({num_of_datasets})!"
        )

    calc = MetricsCalculator()

    method = None
    if args.method == 'causal-rescore':
        method = CausalReScorer(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'simple-causal-rescore':
        method = SimpleCausalReScorer(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'zero-shot-gec':
        method = ZeroShotGec(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'one-shot-gec':
        method = OneShotGec(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'zero-shot-selection':
        method = ZeroShotSelection(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'task-activating-gec':
        method = TaskActivatingGec(args.llm_name, args.tokenizer_name, args.batch_size)
    else:
        raise Exception(f"Method {args.method} is not implemented!")

    eval_df = pd.DataFrame(columns=[
        'results_file',
        'beam_size',
        'alpha',
        'beta',
        'old_wer',
        'new_wer',
        'run_duration',
        'run_duration_in_seconds',
    ])

    num_of_beam_sizes = len(args.beam_sizes)
    grouped_beam_file_paths = [
        args.beams_file_paths[i:i + num_of_beam_sizes] for i in range(0, len(args.beams_file_paths), num_of_beam_sizes)
    ]
    for manifest_file_path, results_dir_path, beams_file_paths in zip(
            args.manifest_file_paths,
            args.results_dir_paths,
            grouped_beam_file_paths
    ):
        Logger.info(f"Processing manifest file {manifest_file_path} ...")
        manifest = ManifestDataset(manifest_file_path)
        ground_truths = manifest.get_transcripts()

        for beams_file_path, beam_size, alpha, beta in zip(beams_file_paths, args.beam_sizes, args.alphas, args.betas):
            Logger.info(f"Processing beam size {beam_size} and beams file {beams_file_path} ...")

            hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
            dataset = HypothesesDataset(hypotheses, ground_truths)

            old_best_hypotheses, old_best_scores, old_best_indices = BestHypothesesSelector.select(dataset)

            results_dictionary = {
                'ground_truth': ground_truths,
                'asr_score': old_best_scores,
                'asr_best_index': old_best_indices,
                'asr_best_hypothesis': old_best_hypotheses,
            }

            used_alpha, used_beta = None, None

            start_time = time.time()
            if isinstance(method, CausalReScorer) or isinstance(method, SimpleCausalReScorer):
                new_scores, used_alpha, used_beta = method.run(dataset, alpha, beta)
                run_duration = time.time() - start_time
                new_best_hypotheses, new_best_scores, new_best_indices = BestHypothesesSelector.select(dataset, new_scores)
                results_dictionary.update({
                    'asr+llm_score': new_best_scores,
                    'asr+llm_best_index': new_best_indices,
                })
            elif (
                    isinstance(method, ZeroShotGec) or
                    isinstance(method, OneShotGec) or
                    isinstance(method, ZeroShotSelection) or
                    isinstance(method, TaskActivatingGec)
            ):
                new_best_hypotheses = method.run(dataset)
                run_duration = time.time() - start_time
            else:
                raise Exception(f"Method {method} is not handled!")

            results_dictionary.update({
                'asr+llm_best_hypothesis': new_best_hypotheses,
            })

            Path(results_dir_path).mkdir(parents=True, exist_ok=True)

            results_file_name = f'{args.llm_name}_{beam_size}'.replace('/', '_')
            results_file_path = f'{results_dir_path}/{results_file_name}.tsv'
            results_df = pd.DataFrame(results_dictionary)
            results_df.to_csv(results_file_path, sep='\t', index=False)
            Logger.info(f"Results saved to {results_file_path}!")

            old_wer_score = calc.calculate_wer(old_best_hypotheses, ground_truths)
            new_wer_score = calc.calculate_wer(new_best_hypotheses, ground_truths)
            new_eval_df = pd.DataFrame({
                'results_file': [results_file_path],
                'beam_size': [beam_size],
                'alpha': [used_alpha],
                'beta': [used_beta],
                'old_wer': [old_wer_score],
                'new_wer': [new_wer_score],
                'run_duration': [str(datetime.timedelta(seconds=run_duration))],
                'run_duration_in_seconds': [round(run_duration, 3)],
            })
            eval_df = pd.concat([eval_df, new_eval_df], ignore_index=True)
            eval_df.to_csv(f'{args.evaluation_dir_path}/evaluation.tsv', sep='\t', index=False)
            Logger.info(eval_df.to_string())
