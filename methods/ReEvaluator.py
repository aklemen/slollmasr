from abc import ABC, abstractmethod
from Hypothesis import Hypothesis
from llms.LargeLanguageModel import LargeLanguageModel


class ReEvaluator(ABC):
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm

    @abstractmethod
    def re_evaluate(self, hypotheses: list[Hypothesis]) -> str:
        pass
