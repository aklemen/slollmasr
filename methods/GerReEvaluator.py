from Hypothesis import Hypothesis
from methods.ReEvaluator import ReEvaluator


class GerReEvaluator(ReEvaluator):
    def re_evaluate(self, hypotheses: list[Hypothesis]) -> str:
        best_hypothesis = self.get_best_hypothesis(hypotheses)
        prompt = "You will perform ASR error correction. You should only output the corrected hypothesis, without any other text.\n"
        prompt += "Hypothesis: " + best_hypothesis["text"]
        print('============================================')
        print(prompt)
        print('============================================')
        return self.llm.prompt(prompt)

    def get_best_hypothesis(self, hypotheses):
        best_hypothesis = None
        for hypothesis in hypotheses:
            if (best_hypothesis is None) or (hypothesis["score"] > best_hypothesis["score"]):
                best_hypothesis = hypothesis
        return best_hypothesis
