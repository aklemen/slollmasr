import json


class Manifest:
    def __init__(self, file_path: str):
        self.transcripts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest_entry = json.loads(line)
                transcript = manifest_entry['text']
                self.transcripts.append(transcript)

    def get_transcripts(self) -> list[str]:
        return self.transcripts