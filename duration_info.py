import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--manifest_file_path', type=str, required=True, help='Path to the manifest file')
args = parser.parse_args()

durations = []
with open(args.manifest_file_path, 'r') as file:
    for line in file:
        entry = json.loads(line)
        durations.append(entry['duration'])

print(f'Total number of audio clips: {len(durations)}')
print(f'Minimum duration: {min(durations)} seconds')
print(f'Maximum duration: {max(durations)} seconds')
print(f'Average duration: {sum(durations) / len(durations)} seconds')
print(f'Sum of all durations in seconds: {sum(durations)}')
print(f'Sum of all durations in hours: {sum(durations) / 3600}')

plt.figure(figsize=(10, 6))
plt.hist(durations, bins=50, edgecolor='black')
plt.title('Distribution of Audio Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()