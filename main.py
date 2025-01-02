import time
from pathlib import Path

import pandas as pd

from BestHypothesesSelector import BestHypothesesSelector
from LargeLanguageModel import LargeLanguageModel
from MetricsCalculator import MetricsCalculator
from Tokenizer import Tokenizer
from methods.CausalReScorer import CausalReScorer
from methods.PromptErrorCorrector import PromptErrorCorrector
from methods.PromptRescorer import PromptRescorer
from parse_args import parse_args
from torch_datasets.HypothesesDataset import HypothesesDataset
from torch_datasets.ManifestDataset import ManifestDataset

if __name__ == '__main__':
    args = parse_args()

    if args.tokenizer_name is None:
        print(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    if args.alphas is None:
        args.alphas = [None] * len(args.beam_sizes)
    if args.betas is None:
        args.betas = [None] * len(args.beam_sizes)

    if len(args.beam_sizes) != len(args.beams_file_paths) or len(args.beam_sizes) != len(args.alphas) or len(args.beam_sizes) != len(args.betas):
        raise ValueError("The number of beam_sizes, alphas and betas should be the same as the number of beams_file_paths")

    Path(args.results_dir_path).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(args.tokenizer_name)
    llm = LargeLanguageModel(args.llm_name)
    calc = MetricsCalculator()

    method = None
    if args.method == 'causal-rescorer':
        method = CausalReScorer(llm, tokenizer)
    elif args.method == 'prompt-rescorer':
        method = PromptRescorer(llm, tokenizer)
    elif args.method == 'prompt-error-corrector':
        method = PromptRescorer(llm, tokenizer)
    else:
        raise Exception(f"Method {args.method} is not implemented!")

    manifest = ManifestDataset(args.manifest_file_path)
    ground_truths = manifest.get_transcripts()

    results_file_basename = f'{args.method}_{args.llm_name}'.replace('/', '_')
    wers_df = pd.DataFrame(columns=['beam_size', 'old_wer', 'new_wer', 'rescoring_duration'])

    for beam_size, beams_file_path, alpha, beta in zip(args.beam_sizes, args.beams_file_paths, args.alphas, args.betas):
        print(f"Processing beam size {beam_size} and beams file {beams_file_path}...")
        hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
        dataset = HypothesesDataset(hypotheses, ground_truths)

        old_best_hypotheses, old_best_scores, old_best_indices = BestHypothesesSelector.select(dataset)

        results_file_name = f'{results_file_basename}_{beam_size}'
        results_dictionary = {
            'ground_truth': ground_truths,
            'asr_score': old_best_scores,
            'asr_best_index': old_best_indices,
            'asr_best_hypothesis': old_best_hypotheses,
        }

        start_time = time.time()
        if isinstance(method, CausalReScorer):
            new_scores, used_alpha, used_beta = method.run(dataset, alpha, beta)
            run_duration = time.time() - start_time
            new_best_hypotheses, new_best_scores, new_best_indices = BestHypothesesSelector.select(dataset, new_scores)
            results_dictionary.update({
                'asr+llm_score': new_best_scores,
                'asr+llm_best_index': new_best_indices,
            })
            results_file_name += f'_alpha={used_alpha}_beta={used_beta}'
        elif isinstance(method, PromptRescorer):
            new_scores = method.run(dataset)
            run_duration = time.time() - start_time
            new_best_hypotheses, new_best_scores, new_best_indices = BestHypothesesSelector.select(dataset, new_scores)
            results_dictionary.update({
                'asr+llm_score': new_best_scores,
                'asr+llm_best_index': new_best_indices,
            })
        elif isinstance(method, PromptErrorCorrector):
            new_best_hypotheses = method.run(dataset)
            run_duration = time.time() - start_time
        else:
            raise Exception(f"Method {method} is not handled!")

        results_dictionary.update({
            'asr+llm_best_hypothesis': new_best_hypotheses,
        })

        results_file_path = f'{args.results_dir_path}/{results_file_name}.tsv'
        results_df = pd.DataFrame(results_dictionary)
        results_df.to_csv(results_file_path, sep='\t', index=False)
        print(f"Results saved to {results_file_path}!")

        old_wer_score = calc.calculate_wer(old_best_hypotheses, ground_truths)
        new_wer_score = calc.calculate_wer(new_best_hypotheses, ground_truths)
        new_wer_df = pd.DataFrame({
            'beam_size': [beam_size],
            'old_wer': [old_wer_score],
            'new_wer': [new_wer_score],
            'run_duration': [run_duration],
        })
        wers_df = pd.concat([wers_df, new_wer_df], ignore_index=True)
        wers_df.to_csv(f'{args.results_dir_path}/{results_file_basename}_wers.tsv', sep='\t', index=False)
        print(wers_df.to_string())
