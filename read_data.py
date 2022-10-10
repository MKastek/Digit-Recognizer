import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, data_path):
        self.train_df = pd.read_csv(data_path)
        self.images = self.train_df.iloc[:, 1:].div(255).values.reshape(len(self.train_df), 1, 28, 28)
        self.img_label = self.train_df.iloc[:, 0].values

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        self.image = torch.tensor(self.images[idx], dtype=torch.float32)
        self.label = torch.tensor(self.img_label[idx])
        return self.image, self.label


class TestDataset(Dataset):

    def __init__(self, data_path):
        self.test_df = pd.read_csv(data_path)
        self.images = self.test_df.iloc[:, :].div(255).values.reshape(len(self.test_df), 1, 28, 28)

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        self.image = torch.tensor(self.images[idx], dtype=torch.float32)
        return self.image


