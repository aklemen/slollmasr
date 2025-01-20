import pandas as pd
from torch.utils.data import Dataset


class HypothesesDataset(Dataset):
    def __init__(self, hypotheses: pd.DataFrame, ground_truths: list[str]):
        self.hypotheses = self._create_hypotheses_df(hypotheses)
        self.ground_truths = ground_truths
        self.beam_size = len(hypotheses) // len(ground_truths)

    def __len__(self):
        return len(self.hypotheses)

    def __getitem__(self, idx):
        hypothesis = self._get_hypothesis_text(idx)
        asr_score = self._get_hypothesis_score(idx)
        return hypothesis, asr_score

    def get_hypotheses_texts(self) -> list[str]:
        return self.hypotheses["text"].tolist()

    def get_hypotheses_scores(self) -> list[float]:
        return self.hypotheses["score"].tolist()

    def get_ground_truths(self) -> list[str]:
        return self.ground_truths

    def get_beam_size(self):
        return self.beam_size

    def get_num_of_samples(self):
        return len(self.ground_truths)

    def _get_hypothesis_text(self, idx):
        return str(self.hypotheses["text"][idx])

    def _get_hypothesis_score(self, idx):
        return self.hypotheses["score"][idx]

    def _create_hypotheses_df(self, hypotheses):
        hypotheses_copy = hypotheses.copy()
        hypotheses["text"].fillna("", inplace=True)
        return hypotheses_copy