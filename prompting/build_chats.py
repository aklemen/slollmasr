from typing import Callable, Union

from utils.logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset

"""
Builds a dataset of prompts in standard or conversational format, depending on the provided build function.
"""
def build_prompt_dataset(
        dataset: HypothesesDataset,
        build_fn: Callable[[list[str]], Union[str, list[dict[str, str]]]],
) -> list[Union[str, list[dict[str, str]]]]:
    samples = []
    for hypotheses in dataset.get_hypotheses_texts_per_sample():
        sample = build_fn(hypotheses)
        samples.append(sample)

    Logger.info(f"Built {len(samples)} samples.")
    return samples
