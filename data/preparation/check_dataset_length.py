import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

def convert_to_standard_format(example):
    return {
        "prompt": example["prompt"][0]["content"],
        "completion": example["completion"][0]["content"],
    }

if __name__ == "__main__":
    dataset = load_dataset('aklemen/whisper-ctc-h2t')['train']
    dataset = dataset.map(convert_to_standard_format)

    model_name = "google/gemma-2b"  # example
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(example):
        text = example["prompt"] + example["completion"]
        tokens = tokenizer(text, truncation=False)["input_ids"]
        example["num_tokens"] = len(tokens)
        return example

    dataset_with_lengths = dataset.map(count_tokens)

    dataset_with_lengths = dataset_with_lengths.sort("num_tokens", reverse=True)

    for entry in dataset_with_lengths.select(range(10)):
        print(entry)
        print("=" * 100)

    print("Average length:", sum(dataset_with_lengths["num_tokens"]) / len(dataset_with_lengths))
    print("Max length:", max(dataset_with_lengths["num_tokens"]))
