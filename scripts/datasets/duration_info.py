import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from logger import Logger

parser = ArgumentParser()
parser.add_argument('--manifest_file_path', type=str, required=True, help='Path to the manifest file')
args = parser.parse_args()

durations = []
with open(args.manifest_file_path, 'r') as file:
    for line in file:
        entry = json.loads(line)
        durations.append(entry['duration'])

Logger.info(f'Total number of audio clips: {len(durations)}')
Logger.info(f'Minimum duration: {min(durations)} seconds')
Logger.info(f'Maximum duration: {max(durations)} seconds')
Logger.info(f'Average duration: {sum(durations) / len(durations)} seconds')
Logger.info(f'Sum of all durations in seconds: {sum(durations)}')
Logger.info(f'Sum of all durations in hours: {sum(durations) / 3600}')

plt.figure(figsize=(10, 6))
plt.hist(durations, bins=50, edgecolor='black')
plt.title('Distribution of Audio Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()