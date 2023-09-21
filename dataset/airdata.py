import torch
from torch.utils.data import Dataset


class AirdataDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = torch.tensor(self.df.iloc[idx, 1], dtype=torch.long)
        return item
