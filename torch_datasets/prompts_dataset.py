from torch.utils.data import Dataset


class PromptsDataset(Dataset):
    def __init__(self, prompts: list[str]):
        self.prompts = prompts
        self.length = len(prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __len__(self):
        return self.length
