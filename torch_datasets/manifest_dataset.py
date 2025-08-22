import json

from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, file_path: str):
        self.transcripts = []
        self.durations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest_entry = json.loads(line)
                self.transcripts.append(manifest_entry['text'])
                self.durations.append(manifest_entry['duration'])

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        return self.transcripts[idx]

    def get_transcripts(self) -> list[str]:
        return self.transcripts

    def get_dataset_duration_in_seconds(self) -> float:
        return sum(self.durations)
