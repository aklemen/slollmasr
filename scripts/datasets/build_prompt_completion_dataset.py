import argparse
import os

import pandas as pd
from datasets import Dataset, concatenate_datasets

from logger import Logger
from torch_datasets.manifest_dataset import ManifestDataset

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_file_path', type=str, required=True)
    parser.add_argument('--ctc_beams_file_path', type=str, required=True)
    parser.add_argument('--whisper_manifest_file_path', type=str, required=True)
    parser.add_argument('--whisper_beams_file_path', type=str, required=True)
    parser.add_argument('--output_dir_path', type=str, required=True)
    parser.add_argument('--whisper_percentage', type=float, default=0.7)

    arguments = parser.parse_args()
    Logger.info("============ ARGUMENTS ============")
    Logger.info(arguments)
    Logger.info("===================================")
    return arguments


def read_grouped_hypotheses(beams_file_path, beam_size):
    hypotheses = pd.read_csv(beams_file_path, delimiter="\t", header=None, names=["text", "score"])
    hypotheses = hypotheses["text"].tolist()
    return [hypotheses[i:i + beam_size] for i in range(0, len(hypotheses), beam_size)]


if __name__ == '__main__':
    args = parse_args()

    Logger.info('Loading transcripts from manifest files ...')
    all_transcripts = ManifestDataset(args.manifest_file_path).get_transcripts()
    all_whisper_transcripts = ManifestDataset(args.whisper_manifest_file_path).get_transcripts()

    if len(set(all_transcripts)) != len(all_transcripts):
        raise ValueError(f"There are duplicate transcripts in the manifest file. Unique transcripts: {len(set(all_transcripts))}, all transcripts: {len(all_transcripts)}.")
    if len(set(all_whisper_transcripts)) != len(all_whisper_transcripts):
        raise ValueError(f"There are duplicate transcripts in the whisper manifest file. Unique transcripts: {len(set(all_whisper_transcripts))}, all transcripts: {len(all_whisper_transcripts)}.")

    Logger.info(f"Loading hypotheses from beams files ...")
    all_ctc_hypotheses = read_grouped_hypotheses(args.ctc_beams_file_path, 10)
    all_whisper_hypotheses = read_grouped_hypotheses(args.whisper_beams_file_path, 10)

    number_of_whisper_samples = round(len(all_transcripts) * args.whisper_percentage)
    number_of_ctc_samples = len(all_transcripts) - number_of_whisper_samples
    Logger.info(f"Number of wanted whisper samples: {number_of_whisper_samples}")
    Logger.info(f"Number of wanted CTC samples: {number_of_ctc_samples}")

    if number_of_whisper_samples > len(all_whisper_transcripts):
        raise ValueError(f"Not enough whisper transcripts available. Requested: {number_of_whisper_samples}, Available: {len(all_whisper_transcripts)}")

    whisper_hypotheses = all_whisper_hypotheses[:number_of_whisper_samples]
    whisper_transcripts = all_whisper_transcripts[:number_of_whisper_samples]
    whisper_dataset = Dataset.from_dict({
        "hypotheses": whisper_hypotheses,
        "ground_truth": whisper_transcripts,
    })

    all_ctc_dataset = Dataset.from_dict({
        "hypotheses": all_ctc_hypotheses,
        "ground_truth": all_transcripts,
    })
    whisper_transcripts_set = set(whisper_transcripts)
    ctc_dataset = all_ctc_dataset.filter(lambda x: x["ground_truth"] not in whisper_transcripts_set)

    Logger.info(f"Whisper dataset: {whisper_dataset}")
    Logger.info(f"CTC dataset: {ctc_dataset}")
    
    dataset = concatenate_datasets([whisper_dataset, ctc_dataset])
    Logger.info(f"Combined dataset: {dataset}")


    prompt_completion_dataset = dataset.map(lambda x: {
        "prompt": [generate_prompt(hypothesis) for hypothesis in x["hypotheses"]],
        "completion": x["ground_truth"]
    }, batched=True, remove_columns=["hypotheses", "ground_truth"])

    Logger.info(f"Prompt-completion dataset: {prompt_completion_dataset}")

    dataset_name = "whisper-ctc-h2t"
    os.makedirs(args.output_dir_path, exist_ok=True)
    output_path = os.path.join(args.output_dir_path, dataset_name)
    prompt_completion_dataset.save_to_disk(output_path)
    prompt_completion_dataset.push_to_hub(f'aklemen/{dataset_name}', private=True)
