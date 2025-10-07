import datetime
import json
from argparse import ArgumentParser

from utils.logger import Logger


def print_duration_stats(durations: list[float]):
    Logger.info(f'Total number of audio clips: {len(durations)}')
    Logger.info(f'Min [s]: {min(durations)}')
    Logger.info(f'Max [s]: {max(durations)}')
    Logger.info(f'Avg [s]: {sum(durations) / len(durations)}')
    Logger.info(f'Sum [s]: {sum(durations)}')
    Logger.info(f'Sum:     {str(datetime.timedelta(seconds=sum(durations)))}')

def display_duration_histogram(durations: list[float]):
    import matplotlib.pyplot as plt
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
        "--manifest_file_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to the manifest files."
    )
    parser.add_argument(
        "--display_histogram",
        action='store_true',
        help="Whether to display a histogram of durations."
    )
    args = parser.parse_args()

    durations_from_manifest = []
    for manifest_path in args.manifest_file_paths:
        with open(manifest_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                durations_from_manifest.append(entry['duration'])

    print_duration_stats(durations_from_manifest)
    if args.display_histogram:
        display_duration_histogram(durations_from_manifest)


if __name__ == "__main__":
    main()
