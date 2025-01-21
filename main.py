import time
from pathlib import Path

import pandas as pd

from BestHypothesesSelector import BestHypothesesSelector
from MetricsCalculator import MetricsCalculator
from methods.CausalReScorer import CausalReScorer
from methods.PipelinePromptErrorCorrector import PipelinePromptErrorCorrector
from parse_args import parse_args
from torch_datasets.HypothesesDataset import HypothesesDataset
from torch_datasets.ManifestDataset import ManifestDataset

if __name__ == '__main__':
    args = parse_args()
    print("=" * 20, " ARGUMENTS ", "=" * 20)
    print(args)
    print("=" * 51)

    if args.tokenizer_name is None:
        print(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
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
    if args.method == 'causal-rescorer':
        method = CausalReScorer(args.llm_name, args.tokenizer_name, args.batch_size)
    elif args.method == 'prompt-error-corrector':
        method = PipelinePromptErrorCorrector(args.llm_name, args.tokenizer_name, args.batch_size)
    else:
        raise Exception(f"Method {args.method} is not implemented!")

    manifest = ManifestDataset(args.manifest_file_path)
    ground_truths = manifest.get_transcripts()

    wers_df = pd.DataFrame(columns=['beam_size', 'alpha', 'beta', 'old_wer', 'new_wer', 'run_duration'])

    num_of_beam_sizes = len(args.beam_sizes)
    grouped_beam_file_paths = [
        args.beams_file_paths[i:i + num_of_beam_sizes] for i in range(0, len(args.beams_file_paths), num_of_beam_sizes)
    ]
    for results_dir_path, beams_file_paths in zip(args.results_dir_paths, grouped_beam_file_paths):
        print(f"Using results directory {results_dir_path} ...")
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)

        for beams_file_path, beam_size, alpha, beta in zip(beams_file_paths, args.beam_sizes, args.alphas, args.betas):
            print(f"Processing beam size {beam_size} and beams file {beams_file_path} ...")

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
            if isinstance(method, CausalReScorer):
                new_scores, used_alpha, used_beta = method.run(dataset, alpha, beta)
                run_duration = time.time() - start_time
                new_best_hypotheses, new_best_scores, new_best_indices = BestHypothesesSelector.select(dataset,
                                                                                                       new_scores)
                results_dictionary.update({
                    'asr+llm_score': new_best_scores,
                    'asr+llm_best_index': new_best_indices,
                })
            elif isinstance(method, PipelinePromptErrorCorrector):
                new_best_hypotheses = method.run(dataset)
                run_duration = time.time() - start_time
            else:
                raise Exception(f"Method {method} is not handled!")

            results_dictionary.update({
                'asr+llm_best_hypothesis': new_best_hypotheses,
            })

            results_file_name = f'{args.llm_name}_{beam_size}'.replace('/', '_')
            results_file_path = f'{results_dir_path}/{results_file_name}.tsv'
            results_df = pd.DataFrame(results_dictionary)
            results_df.to_csv(results_file_path, sep='\t', index=False)
            print(f"Results saved to {results_file_path}!")

            old_wer_score = calc.calculate_wer(old_best_hypotheses, ground_truths)
            new_wer_score = calc.calculate_wer(new_best_hypotheses, ground_truths)
            new_wer_df = pd.DataFrame({
                'beam_size': [beam_size],
                'alpha': [used_alpha],
                'beta': [used_beta],
                'old_wer': [old_wer_score],
                'new_wer': [new_wer_score],
                'run_duration': [run_duration],
            })
            wers_df = pd.concat([wers_df, new_wer_df], ignore_index=True)
            wers_df.to_csv(f'{args.evaluation_dir_path}/wers.tsv', sep='\t', index=False)
            print(wers_df.to_string())
