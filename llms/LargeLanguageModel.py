from abc import ABC, abstractmethod


class LargeLanguageModel(ABC):
    @abstractmethod
    def prompt(self, input_text: str) -> str:
        pass

    @abstractmethod
    def score(self, input_text: str) -> float:
        pass
