from abc import ABC, abstractmethod

from torch_datasets.HypothesesDataset import HypothesesDataset


class Method(ABC):
    @abstractmethod
    def run(self, dataset: HypothesesDataset):
        pass
