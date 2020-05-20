import numpy as np
import torch

from torch.utils.data import Dataset


class HeadlineDataset(Dataset):
    def __init__(self, articles, titles, batch_size):
        self.in_data = np.array(articles, dtype=np.int)
        self.label = np.array(titles, dtype=np.int)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.in_data) // self.batch_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.in_data)
        y = torch.from_numpy(self.label)

        return x[idx], y[idx]

