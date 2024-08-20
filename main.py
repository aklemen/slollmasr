import logging
import time
import pandas as pd

from MetricsCalculator import MetricsCalculator
from methods.CausalReScorer import CausalReScorer
from HypothesesDataset import HypothesesDataset
from Manifest import Manifest
from llms.LargeLanguageModel import LargeLanguageModel
from BestHypothesesSelector import BestHypothesesSelector

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_time = time.time()

    hypotheses = pd.read_csv("/hf-home/beams_output/preds_out_width5_alpha1.0_beta0.0.tsv", delimiter="\t", header=None, names=["text", "score"])
    manifest = Manifest("/dataset/artur/v1.0/nemo/test.nemo")
    ground_truths = manifest.get_transcripts()

    hf_llms = [
        "openai-community/gpt2",
        "meta-llama/Meta-Llama-3.1-8B",
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    llm = LargeLanguageModel(hf_llms[1])
    rescorer = CausalReScorer(llm)

    dataset = HypothesesDataset(hypotheses, ground_truths, llm.tokenizer, 5, 512)

    new_scores = rescorer.re_score(dataset)

    old_best = BestHypothesesSelector.select(dataset)
    new_best = BestHypothesesSelector.select(dataset, new_scores)

    calc = MetricsCalculator()
    old_wer_score = calc.calculate_wer(predictions=old_best, references=ground_truths)
    new_wer_score = calc.calculate_wer(predictions=new_best, references=ground_truths)

    print(f"Scores length: {len(new_scores)}")
    print(f"First 5 score: {new_scores[:5]}")
    print(f"First 5 hypothesis: {new_best[:5]}")
    print(f"Old WER: {old_wer_score}")
    print(f"New WER: {new_wer_score}")
    print(f"Execution time: {time.time() - start_time} seconds")
