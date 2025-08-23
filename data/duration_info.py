import datetime
import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from utils.logger import Logger


def print_duration_stats(durations: list[float]):
    Logger.info(f'Total number of audio clips: {len(durations)}')
    Logger.info(f'Minimum duration: {min(durations)} seconds')
    Logger.info(f'Maximum duration: {max(durations)} seconds')
    Logger.info(f'Average duration: {sum(durations) / len(durations)} seconds')
    Logger.info(f'Sum of all durations in seconds: {sum(durations)}')
    Logger.info(f'Sum of all durations formatted: {str(datetime.timedelta(seconds=sum(durations)))}')

def display_duration_histogram(durations: list[float]):
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, edgecolor='black')
    plt.title('Distribution of Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest_file_path",
        type=str,
        required=True,
        help="Path to the manifest file."
    )
    args = parser.parse_args()

    durations_from_manifest = []
    with open(args.manifest_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            durations_from_manifest.append(entry['duration'])

    print_duration_stats(durations_from_manifest)
    display_duration_histogram(durations_from_manifest)


if __name__ == "__main__":
    main()