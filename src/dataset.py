import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
from sklearn.model_selection import train_test_split


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

class CS2PredictionDataset(Dataset):
    def __init__(self, data_root = f"../../game_demos"):
        
        self.csv_file_out = "preprocessed_data.csv"
        self.data = []
        self.labels = []

        self.dataset = None

        self.max_event_length = 0
        
        self.data_root = data_root


        

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
       
        return data, label
    
    def __len__(self):
        return len(self.dataset)