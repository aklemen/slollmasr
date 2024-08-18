import logging
import time
import pandas as pd

from methods.ReScorer import ReScorer
from HypothesesDataset import HypothesesDataset
from Manifest import Manifest
from llms.LargeLanguageModel import LargeLanguageModel


# def read_transcripts(json_file_path: str) -> list[Transcript]:
#     with open(json_file_path, 'r') as file:
#         json_data = json.load(file)
#
#     transcripts = []
#
#     for item in json_data:
#         transcript = Transcript(item["hypotheses"], item["transcript"])
#         transcripts.append(transcript)
#
#     return transcripts
#
#
# def run_reevaluation(re_evaluator: ReEvaluator):
#     transcripts = read_transcripts("hypotheses_sample.json")
#     for transcript in transcripts:
#         re_evaluated_transcript = re_evaluator.re_evaluate(transcript.get_hypotheses())
#         print(re_evaluated_transcript)
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_time = time.time()
    
    hypotheses = pd.read_csv("./nemo_hypotheses_beam_size_5.tsv", delimiter="\t", header=None, names=["text", "score"])
    manifest = Manifest("./nemo_manifest.nemo")
    ground_truths = manifest.get_transcripts()
    
    hf_llms = [
        "openai-community/gpt2",
        "meta-llama/Meta-Llama-3.1-8B",
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base"
    ]
    llm = LargeLanguageModel(hf_llms[0])
    rescorer = ReScorer(llm)

    dataset = HypothesesDataset(hypotheses, ground_truths, llm.tokenizer, 5, 256)
    
    new_scores = rescorer.re_score(dataset)
    
    print(f"Scores length: {len(new_scores)}")
    print(f"First score: {new_scores[0]}")
    print(f"Execution time: {time.time() - start_time} seconds")
