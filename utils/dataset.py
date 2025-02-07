from utils.word_tokenizer import wordTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch


def create_datasets(
    trn_ratio=0.8,
    val_ratio=0.1,
    tst_ratio=0.1,
):
    df = pd.read_csv(r"datasets/normalized.csv")
    trn = df.sample(frac=trn_ratio)
    df = df.drop(trn.index)
    df.reset_index(drop=True, inplace=True)

    val_ratio = val_ratio / (val_ratio + tst_ratio)

    val = df.sample(frac=val_ratio)
    tst = df.drop(val.index)

    trn.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    tst.reset_index(drop=True, inplace=True)

    trn = custom_dataset(trn)
    val = custom_dataset(val)
    tst = custom_dataset(tst)
    return trn, val, tst


class custom_dataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.input_val = self.data.iloc[:, 1]
        self.target_word = self.data.iloc[:, 0]
        self.tokenizer = wordTokenizer()

    def __len__(self):
        return len(self.input_val)

    def __getitem__(self, idx):
        input_val = self.input_val[idx]
        input_val = torch.tensor(input_val)
        target_word = self.target_word[idx]
        return input_val, target_word
