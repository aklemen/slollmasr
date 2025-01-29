from typing import Callable

from logger import Logger
from torch_datasets.hypotheses_dataset import HypothesesDataset

def build_chats(
        dataset: HypothesesDataset,
        build_fn: Callable[[list[str]], list[dict[str, str]]],
) -> list[list[dict[str, str]]]:
    chats = []
    for hypotheses in dataset.get_hypotheses_texts_per_sample():
        chat = build_fn(hypotheses)
        chats.append(chat)

    Logger.info(f"Built {len(chats)} chats.")
    return chats
