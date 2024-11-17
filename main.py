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
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--beams_file_path', type=str, required=True)
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beam_size', type=int, required=True)
    parser.add_argument('--results_dir_path', type=str, required=True)
    parser.add_argument('--log', type=bool, default=False)
    args = parser.parse_args()

    if args.log:
        logging.getLogger().setLevel(logging.INFO)

    if args.tokenizer_name is None:
        logging.info(f"Tokenizer name was not given, using LLM name '{args.llm_name}'")
        args.tokenizer_name = args.llm_name

    start_time = time.time()

    hypotheses = pd.read_csv(args.beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    manifest = ManifestDataset(args.manifest_file_path)
    ground_truths = manifest.get_transcripts()

    tokenizer = Tokenizer(args.tokenizer_name)
    llm = LargeLanguageModel(args.llm_name)
    dataset = HypothesesDataset(hypotheses, tokenizer, args.beam_size, 512)
    rescorer = CausalReScorer(llm)

    new_scores = rescorer.re_score(dataset)

    old_best_hypotheses, old_best_scores = BestHypothesesSelector.select(dataset)
    new_best_hypotheses, new_best_scores = BestHypothesesSelector.select(dataset, new_scores)

    calc = MetricsCalculator()
    old_wer_score = calc.calculate_wer(old_best_hypotheses, ground_truths)
    new_wer_score = calc.calculate_wer(new_best_hypotheses, ground_truths)

    Path(args.results_dir_path).mkdir(parents=True, exist_ok=True)
    out_file_name = f'rescored_{args.llm_name}_{args.beam_size}.tsv'.replace('/', '_')
    out_file = f'{args.results_dir_path}/{out_file_name}'
    data = {
        'ground_truth': ground_truths,
        'asr_best_hypothesis': old_best_hypotheses,
        'asr_llm_best_hypothesis': new_best_hypotheses,
        'asr_score': old_best_scores,
        'asr_llm_score': new_best_scores
    }

    data = df = pd.DataFrame(data)
    df.to_csv(out_file, sep='\t', index=False)

    print(f"Old WER: {old_wer_score}")
    print(f"New WER: {new_wer_score}")
    print(f"Execution time: {time.time() - start_time} seconds")
    print('ground_truths: ', ground_truths[0])
    print('old_best_hypotheses: ', old_best_hypotheses[0])
    print('new_best_hypotheses: ', new_best_hypotheses[0])
    print('old_best_scores: ', old_best_scores[0])
    print('new_best_scores: ', new_best_scores[0])
