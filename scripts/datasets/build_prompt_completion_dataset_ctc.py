import argparse
import json
import os

import pandas as pd
from datasets import Dataset

from logger import Logger

prompt_template = ("### Navodilo:\n"
                   "Spodaj je najboljša hipoteza, ki jo je za avdio posnetek generiral sistem za razpoznavanje govora. "
                   "Preglej jo in jo s pomočjo ostalih hipotez popravi, če je potebno. "
                   "Potem izpiši končni transkript.\n\n"
                   "### Najboljša hipoteza:\n{best_hypothesis}\n\n"
                   "### Ostale hipoteze:\n{other_hypotheses}\n\n"
                   f"### Transkript:\n")


def generate_prompt(hypotheses: list[str]):
    return prompt_template.format_map({
        "best_hypothesis": hypotheses[0],
        "other_hypotheses": "\n".join(hypotheses[1:]),
    })


def generate_sample(example):
    return {
        "prompt": [{ "role": "user", "content": generate_prompt(example["hypotheses"]) }],
        "completion": [{ "role": "assistant", "content": example["ground_truth"] }]
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--beams_file_path', type=str, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)

    arguments = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(arguments)
    Logger.info("===================================")
    return arguments


def read_manifest(manifest_file_path):
    with open(manifest_file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def read_grouped_hypotheses(beams_file_path, beam_size):
    hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    hypotheses = hypotheses["text"].tolist()
    hypotheses = [str(text) for text in hypotheses]
    return [hypotheses[i:i + beam_size] for i in range(0, len(hypotheses), beam_size)]


if __name__ == '__main__':
    args = parse_args()

    Logger.info('Loading transcripts from manifest files ...')
    ctc_manifest = read_manifest(args.manifest_file_path)

    Logger.info(f"Loading hypotheses from beams files ...")
    ctc_hypotheses = read_grouped_hypotheses(args.beams_file_path, 10)

    dataset = Dataset.from_dict({
        "hypotheses": ctc_hypotheses,
        "ground_truth": [entry["text"] for entry in ctc_manifest],
    })

    prompt_completion_dataset = dataset.map(generate_sample, remove_columns=["hypotheses", "ground_truth"])

    Logger.info(f"Prompt-completion dataset: {prompt_completion_dataset}")

    dataset_name = "ctc-h2t"
    os.makedirs(args.output_dir_path, exist_ok=True)
    output_path = os.path.join(args.output_dir_path, dataset_name)
    prompt_completion_dataset.save_to_disk(output_path)
    prompt_completion_dataset.push_to_hub(f'aklemen/{dataset_name}', private=True)
