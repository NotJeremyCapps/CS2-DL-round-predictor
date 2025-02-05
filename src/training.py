
#have to initialize hidden state of model in model

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import random

from dataset import CS2PredictionDataset

from model import CS2LSTM

# torch.set_printoptions(profile="full")


class ModelTrainer():

    def __init__(self):

        self.epochs = 20 

        self.model = CS2LSTM()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers

        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.data

        # self.model = self.lstm

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Wrap model for multi-GPU

        # hidden = self.model.hidden_init() #initialize hidden variable

        self.model.to(self.device)  # Move model to GPUs


        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.dataset = CS2PredictionDataset(list="../game_demos/preprocessed/de_anubis/rounds.txt", sequence_length=30)

        # self.training_data, self.testing_data = torch.utils.data.random_split(data_set, [0.75, 0.25])

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=3,
                                     num_workers=0
                                     )



    def train(self): #loss function and optimizer should be prebuilt functions

        self.model.train() #inherited function from model.py, sets model in training modes

        for i, (target, x_main_data, x_data_weap, new_round) in enumerate(self.dataloader):
            with open('passedtotraining_seq3.txt', 'a') as f:
                print(f"Target: {target}, Shape: {target.size()};\n Main_data: {x_main_data}, Shape: {x_main_data.size()};\n Weapon_data: {x_data_weap}, Shape: {x_data_weap.size()};\n New_Round: {new_round}, Shape: {new_round.shape};\n\n\n", file = f)
    
            out, hidden = self.model(x_main_data, x_data_weap, hidden) #hidden is info from past
            loss = self.lossFunc(out, x)

            self.optimizer.zero_grad() #zeros grad from previous run
            self.lossFunc.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
            self.optimizer.step()#changes weights and bias of parameters based on loss
            if True in new_round:
                hidden = self.model.hidden_reset()#whatever the function is to reset hidden
        


    def test(self):

        self.model.eval()

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


    def train_model(self):

        for epoch in range(self.epochs):
            self.train()



mt = ModelTrainer()

mt.train_model()