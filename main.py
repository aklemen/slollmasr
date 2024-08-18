import json

from HypothesesDataset import HypothesesDataset
from methods.ReEvaluator import ReEvaluator
from Transcript import Transcript
from methods.ReScorer import ReScorer


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
    # llm = Gpt2LargeLanguageModel()
    # method = GerReEvaluator(llm)
    # run_reevaluation(method)
    rescorer = ReScorer("openai-community/gpt2")
    dataset = HypothesesDataset("./nemo_hypotheses_beam_size_5.tsv", "./nemo_manifest.nemo", rescorer.tokenizer, 5, 256)
    new_scores = rescorer.re_score(dataset)
    print(new_scores)
