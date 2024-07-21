from llms.Gpt2LargeLanguageModel import Gpt2LargeLanguageModel
from methods.GerReEvaluator import GerReEvaluator
from methods.ReEvaluator import ReEvaluator
from Transcript import Transcript
import json


def read_transcripts(json_file_path: str) -> list[Transcript]:
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    transcripts = []

    for item in json_data:
        transcript = Transcript(item["hypotheses"], item["transcript"])
        transcripts.append(transcript)

    return transcripts


def run_reevaluation(re_evaluator: ReEvaluator):
    transcripts = read_transcripts("hypotheses_sample.json")
    for transcript in transcripts:
        re_evaluated_transcript = re_evaluator.re_evaluate(transcript.get_hypotheses())
        print(re_evaluated_transcript)
    

if __name__ == '__main__':
    llm = Gpt2LargeLanguageModel()
    method = GerReEvaluator(llm)
    run_reevaluation(method)
