from abc import ABC, abstractmethod

from LargeLanguageModel import LargeLanguageModel
from Tokenizer import Tokenizer
from torch_datasets.HypothesesDataset import HypothesesDataset


class Method(ABC):
    def __init__(self, llm: LargeLanguageModel, tokenizer: Tokenizer):
        pass

    @abstractmethod
    def run(self, dataset: HypothesesDataset):
        pass
