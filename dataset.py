import numpy as np
import torch

from torch.utils.data import Dataset


class TitleDataset(Dataset):
    def __init__(self, articles, titles, article_len, title_len):
        self.in_data = articles
        self.label = titles
        self.article_len = article_len
        self.title_len = title_len

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, item):
        pass

