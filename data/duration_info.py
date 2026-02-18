import datetime
import json
from argparse import ArgumentParser
import numpy as np


def print_duration_stats(durations: list[float]):
    print(f'Total number of audio clips: {len(durations)}')
    print(f'Min [s]: {min(durations)}')
    print(f'Max [s]: {max(durations)}')
    print(f'Avg [s]: {sum(durations) / len(durations)}')
    print(f'Sum [s]: {sum(durations)}')
    print(f'Sum [h]: {sum(durations) / 3600}')
    print(f'Sum:     {str(datetime.timedelta(seconds=sum(durations)))}')

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

    for manifest_path in args.manifest_file_paths:
        durations_from_manifest = []
        with open(manifest_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                durations_from_manifest.append(entry['duration'])
        print(f"================= {manifest_path} ================= ")
        print_duration_stats(durations_from_manifest)

        # split lengths into bins with numpy and print the counts
        # the bins are 0-10, 10-20, 20-30, ...
        bins = np.arange(0, max(durations_from_manifest) + 5, 5)
        counts, _ = np.histogram(durations_from_manifest, bins=bins)
        for i in range(len(bins) - 1):
            print(f"{bins[i]:>5} - {bins[i+1]:>5} s: {counts[i]}")

        if args.display_histogram:
            display_duration_histogram(durations_from_manifest)


if __name__ == "__main__":
    main()
