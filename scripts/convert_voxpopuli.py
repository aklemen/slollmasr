import argparse
import csv
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tarfile
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


def process_files(csv_file, data_root):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to data_root.

    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        data_root: str, path to dir to save results; wav/ dir will be created
    """
    wav_dir = os.path.join(data_root, 'wav/')
    os.makedirs(wav_dir, exist_ok=True)
    audio_clips_path = os.path.dirname(csv_file) + '/audio/'

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

    logging.info('Converting ogg to wav for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        # next(reader, None)  # skip the headers
        data = []
        for row in reader:
            file_name = row['id']
            # add the mp3 extension if the tsv entry does not have it
            if not file_name.endswith('.ogg'):
                file_name += '.ogg'
            data.append((file_name, row['normalized_text'], row['raw_text']))
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            data = list(tqdm(pool.imap(process, data), total=len(data)))
    return data


def main():
    data_root = "./"
    os.makedirs(data_root, exist_ok=True)

    for csv_file in ["asr_dev.tsv", "asr_test.tsv", "asr_train.tsv"]:
        data = process_files(
            csv_file=os.path.join(data_root, csv_file),
            data_root=os.path.join(data_root, os.path.splitext(csv_file)[0]),
        )
        logging.info('Creating manifests...')
        create_manifest(
            data=data,
            output_name=f'{os.path.splitext(csv_file)[0]}_manifest.json',
            manifest_path=data_root,
        )


if __name__ == "__main__":
    main()
