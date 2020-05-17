import numpy as np

from torch.utils.data import Dataset


class TitleDataset(Dataset):
    def __init__(self, articles, titles):
        self.in_data = articles
        self.label = titles

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, item):
        pass
