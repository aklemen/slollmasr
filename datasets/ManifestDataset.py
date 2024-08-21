import json

from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, file_path: str):
        self.transcripts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest_entry = json.loads(line)
                transcript = manifest_entry['text']
                self.transcripts.append(transcript)

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        return self.transcripts[idx]

    def get_transcripts(self) -> list[str]:
        return self.transcripts
