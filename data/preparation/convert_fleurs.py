import argparse
import csv
import json
import logging
import multiprocessing
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

import sox
from sox import Transformer
from tqdm import tqdm

def create_manifest(data: List[tuple], output_name: str, manifest_path: str):
    output_file = Path(manifest_path) / output_name
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with output_file.open(mode='w') as f:
        for wav_path, duration, text, raw_text in tqdm(data, total=len(data)):
            if wav_path != '':
                # skip invalid input files that could not be converted
                f.write(
                    json.dumps({'audio_filepath': os.path.abspath(wav_path), "duration": duration, 'text': text, 'raw_text': raw_text}, ensure_ascii=False)
                    + '\n'
                )


def process_files(csv_file, data_root, tp):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to data_root.

    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        data_root: str, path to dir to save results; wav/ dir will be created
    """
    wav_dir = os.path.join(data_root, 'wav/')
    os.makedirs(wav_dir, exist_ok=True)
    audio_clips_path = os.path.dirname(csv_file) + '/audio/' + tp

    def process(x):
        file_path, text, raw_text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.lower().strip()
        audio_path = os.path.join(audio_clips_path, file_path)
        if os.path.getsize(audio_path) == 0:
            logging.warning(f'Skipping empty audio file {audio_path}')
            return '', '', ''

        output_wav_path = os.path.join(wav_dir, file_name + '.wav')

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=16000)
            tfm.channels(n_channels=1)
            tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

        duration = sox.file_info.duration(output_wav_path)
        return output_wav_path, duration, text, raw_text

    logging.info('Converting wav to wav for {}.'.format(csv_file))
    fieldnames = ['id', 'file_name', 'raw_text', 'normalized_text', 'a', 'b', 'c']
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=fieldnames)
        # next(reader, None)  # skip the headers
        data = []
        for row in reader:
            print(row)
            file_name = row['file_name']
            # add the mp3 extension if the tsv entry does not have it
            if not file_name.endswith('.wav'):
                file_name += '.wav'
            data.append((file_name, row['normalized_text'], row['raw_text']))
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            data = list(tqdm(pool.imap(process, data), total=len(data)))
    return data


def main():
    data_root = "./"
    os.makedirs(data_root, exist_ok=True)

    for csv_file in ["dev.tsv", "test.tsv", "train.tsv"]:
        data = process_files(
            csv_file=os.path.join(data_root, csv_file),
            data_root=os.path.join(data_root, os.path.splitext(csv_file)[0]),
            tp=os.path.splitext(csv_file)[0]
        )
        logging.info('Creating manifests...')
        create_manifest(
            data=data,
            output_name=f'{os.path.splitext(csv_file)[0]}.json',
            manifest_path=data_root,
        )


if __name__ == "__main__":
    main()
