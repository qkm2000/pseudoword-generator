from utils.word_tokenizer import wordTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch


class trainDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(r"datasets/normalized.csv", nrows=100, skiprows=1)
        self.input_val = df.iloc[:, 1]
        self.target_word = df.iloc[:, 0]
        self.tokenizer = wordTokenizer()

    def __len__(self):
        return len(self.input_val)

    def __getitem__(self, idx):
        input_val = self.input_val[idx]
        input_val = torch.tensor(input_val)
        target_word = self.target_word[idx]
        return input_val, target_word


class valDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(r"datasets/normalized.csv", skiprows=100, nrows=10)
        self.input_val = df.iloc[:, 1]
        self.target_word = df.iloc[:, 0]
        self.tokenizer = wordTokenizer()

    def __len__(self):
        return len(self.input_val)

    def __getitem__(self, idx):
        input_val = self.input_val[idx]
        input_val = torch.tensor(input_val)
        target_word = self.target_word[idx]
        return input_val, target_word


class testDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(r"datasets/normalized.csv", skiprows=110)
        self.input_val = df.iloc[:, 1]
        self.target_word = df.iloc[:, 0]
        self.tokenizer = wordTokenizer()

    def __len__(self):
        return len(self.input_val)

    def __getitem__(self, idx):
        input_val = self.input_val[idx]
        input_val = torch.tensor(input_val)
        target_word = self.target_word[idx]
        return input_val, target_word
