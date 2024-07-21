from Hypothesis import Hypothesis


class Transcript:
    def __init__(self, hypotheses: list[Hypothesis], ground_truth: str):
        self._ground_truth = ground_truth
        self._hypotheses = hypotheses

    def get_ground_truth(self) -> str:
        return self._ground_truth

    def get_hypotheses(self) -> list[Hypothesis]:
        return self._hypotheses
