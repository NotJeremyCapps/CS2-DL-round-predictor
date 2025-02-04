
#have to initialize hidden state of model in model

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random


class ModelTrainer():

    def __init__(self, input_size, hidden_size, num_layers, output_size, data_set):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.lstm

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Wrap model for multi-GPU

        self.model.to(device)  # Move model to GPUs


        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.training_data, self.testing_data = torch.utils.data.random_split(data_set, [0.75, 0.25])



    def train_model(self): #loss function and optimizer should be prebuilt functions
        for batch in self.training_data:
            with open('passedtotraining.txt', 'a') as f:
                print(batch, file = f)
        '''
        hidden = self.model.hidden_init() #initialize hidden variable
        self.model.train() #inherited function from model.py, sets model in training mode

        for x, y, z, new_round_flag in self.training_data:#x =target, y = main data, z = enums data, data_loader has the entire dataset
            out, hidden = self.model(y,z, hidden) #hidden is info from past
            loss = self.lossFunc(out, x)

            self.optimizer.zero_grad() #zeros grad from previous run
            loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
            self.optimizer.step()#changes weights and bias of parameters based on loss
            if new_round_flag:
                hidden = self.model.hidden_reset()#whatever the function is to reset hidden
        '''


    def test_model(self):
        batches = len(self.testing_data)
        hidden = self.model.hidden_init() #initialize hidden variable
        self.model.eval()
        loss = 0

        with torch.no_grad(): #stops gradient computation - it is unnecessary
            for x, y, z in self.testing_data:
                out, hidden = self.model(y,z, hidden)
                loss += self.lossFunc(out, x).item()

        final_loss = loss/batches
        print("Final Lost in Test:", final_loss)


