from argparse import ArgumentParser
import logging
from pathlib import Path
import time
import pandas as pd

from MetricsCalculator import MetricsCalculator
from Tokenizer import Tokenizer
from methods.CausalReScorer import CausalReScorer
from torch_datasets.HypothesesDataset import HypothesesDataset
from torch_datasets.ManifestDataset import ManifestDataset
from LargeLanguageModel import LargeLanguageModel
from BestHypothesesSelector import BestHypothesesSelector

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=False)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_paths', nargs='+', type=str, required=True)
    parser.add_argument('--beam_sizes', nargs='+', type=int, required=True)
    parser.add_argument('--alphas', nargs='+', type=int, required=False)
    parser.add_argument('--betas', nargs='+', type=int, required=False)
    parser.add_argument('--results_dir_path', type=str, required=True)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    if len(args.beam_sizes) != len(args.beams_file_paths):
        raise ValueError("The number of beam_sizes should be the same as the number of beams_file_paths")

    if args.log:
        logging.getLogger().setLevel(logging.INFO)

    if args.tokenizer_name is None:
        logging.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    Path(args.results_dir_path).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(args.tokenizer_name)
    llm = LargeLanguageModel(args.llm_name)
    rescorer = CausalReScorer(llm)
    calc = MetricsCalculator()

    manifest = ManifestDataset(args.manifest_file_path)
    ground_truths = manifest.get_transcripts()

    for beam_size, beams_file_path, alpha, beta in zip(args.beam_sizes, args.beams_file_paths, args.alphas, args.betas):
        print(f"Processing beam size {beam_size} and beams file {beams_file_path}...")
        print(f"Alpha: {alpha}, Beta: {beta}")
        hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
        dataset = HypothesesDataset(hypotheses, ground_truths, tokenizer, beam_size, 512)
        start_time = time.time()
        new_scores = rescorer.re_score(dataset, alpha, beta)
        rescoring_duration = time.time() - start_time

        old_best_hypotheses, old_best_scores, old_best_indices = BestHypothesesSelector.select(dataset)
        new_best_hypotheses, new_best_scores, new_best_indices = BestHypothesesSelector.select(dataset, new_scores)
        old_wer_score = calc.calculate_wer(old_best_hypotheses, ground_truths)
        new_wer_score = calc.calculate_wer(new_best_hypotheses, ground_truths)

        file_name = f'{args.llm_name}_{beam_size}_alpha={alpha}_beta={beta}'.replace('/', '_')

        results_file_path = f'{args.results_dir_path}/{file_name}.tsv'
        data = df = pd.DataFrame({
            'ground_truth': ground_truths,
            'asr_best_hypothesis': old_best_hypotheses,
            'asr_llm_best_hypothesis': new_best_hypotheses,
            'asr_score': old_best_scores,
            'asr_llm_score': new_best_scores,
            'asr_best_index': old_best_indices,
            'asr_llm_best_index': new_best_indices,
        })
        df.to_csv(results_file_path, sep='\t', index=False)

        print(f"Results saved to {results_file_path}!")

        log_file_path = f'{args.results_dir_path}/{file_name}.log'
        with(open(log_file_path, 'w')) as log_file:
            log_file.write(f"Old WER: {old_wer_score}\n")
            log_file.write(f"New WER: {new_wer_score}\n")
            log_file.write(f"Rescoring duration: {rescoring_duration} seconds\n")

        print(f"Old WER: {old_wer_score}")
        print(f"New WER: {new_wer_score}")
        print(f"Rescoring duration: {rescoring_duration} seconds")