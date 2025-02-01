
#have to initialize hidden state of model in model

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



def train_model(data_loader, model, loss_function, optimizer): #loss function and optimizer should be prebuilt functions
    hidden = model.hidden_init() #initialize hidden variable
    model.train() #inherited function from model.py, sets model in training mode

    for x, y, z, new_round_flag in data_loader:#x =target, y = main data, z = enums data, data_loader has the entire dataset
        out, hidden = model(y,z, hidden) #hidden is info from past
        loss = loss_function(out, x)

        optimizer.zero_grad() #zeros grad from previous run
        loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
        optimizer.step()#changes weights and bias of parameters based on loss
        if new_round_flag:
            hidden = model.hidden_reset()#whatever the function is to reset hidden


def test_model(data_loader, model, loss_function):
    batches = len(data_loader)
    hidden = model.hidden_init() #initialize hidden variable
    model.eval()
    loss = 0

    with torch.no_grad(): #stops gradient computation - it is unnecessary
        for x, y, z in data_loader:
            out, hidden = model(y,z, hidden)
            loss += loss_function(out, x).item()

    final_loss = loss/batches
    print("Final Lost in Test:", final_loss)


