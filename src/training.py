import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random
import CS2PredictionDataset


class ModelTrainer():

    def __init__(self, input_size, hidden_size, num_layers, output_size, data_set):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.lstm

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)  # Wrap model for multi-GPU

        model.to(device)  # Move model to GPUs


        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.training_data, self.testing_data = torch.utils.data.random_split(data_set, [0.75, 0.25])


#def train_model(data_loader, model, loss_function, optimizer):
#