from typing import Tuple
from evaluate import load
class MetricsCalculator:
    def __init__(self):
        self.cer = load("cer")
        self.wer = load("wer")

    def calculate_wer(self, predictions: list[str], references: list[str]) -> float:
        return self.wer.compute(predictions, references)

    def calculate_cer(self, predictions: list[str], references: list[str]) -> float:
        return self.cer.compute(predictions, references)

    def calculate_wer_and_cer(self, predictions: list[str], references: list[str]) -> tuple[float, float]:
        w = self.calculate_wer(predictions, references)
        c = self.calculate_cer(predictions, references)
        return w, c
