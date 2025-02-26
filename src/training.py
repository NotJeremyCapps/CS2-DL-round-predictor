
#have to initialize hidden state of model in model

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os


from dataset import CS2PredictionDataset, split_dataset

from model import CS2LSTM

# torch.set_printoptions(profile="full")

DEMO_PATH = "../../demos/"


class ModelTrainer():

    def __init__(self):

        self.total_epochs = 20 

        self.batch_size = 28

        self.seq_len = 60



        self.model = CS2LSTM(n_feature=None, out_feature=1,n_hidden=256,n_layers=2)

        self.hidden = self.model.init_hidden(self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Wrap model for multi-GPU

        # hidden = self.model.hidden_init() #initialize hidden variable

        self.model.to(self.device)  # Move model to GPUs


        self.lossFunc = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.split_dataset = True

        if self.split_dataset:
            split_dataset(0.8, os.path.join(DEMO_PATH, "preprocessed"))

        self.trainset = CS2PredictionDataset(list=os.path.join(DEMO_PATH, "/preprocessed/rounds_train.txt"), sequence_length=self.seq_len)
        self.testset = CS2PredictionDataset(list=os.path.join(DEMO_PATH, "/preprocessed/rounds_test.txt"), sequence_length=self.seq_len)

        print(f"Trainset len : {len(self.trainset)}")
        print(f"Testset len : {len(self.testset)}")

        # self.training_data, self.testing_data = torch.utils.data.random_split(data_set, [0.75, 0.25])

        self.train_loader = DataLoader(dataset=self.trainset,
                                     batch_size=self.batch_size,
                                     num_workers=0,
                                     drop_last=True
                                     )
        
        self.test_loader = DataLoader(dataset=self.testset,
                                     batch_size=self.batch_size,
                                     num_workers=0,
                                     drop_last=True
                                     )

        log_dir = 'logs/' + datetime.now().strftime('%B%d_%H_%M_%S')
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, global_tag, metric_dict, global_step):

        for tag, value in metric_dict.items():
            self.writer.add_scalar(f"{global_tag}/{tag}", value, global_step)

    def train(self, epoch): #loss function and optimizer should be prebuilt functions

        self.model.train() #inherited function from model.py, sets model in training modes

        metric_dict = {}
        total = 0
        correct = 0
        num_preds = 0

        # hidden = None
        with tqdm(self.train_loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (target, new_round, x_main_data, x_prim_weap, x_sec_weap) in enumerate(tepoch):
                
                # print(f"Len of main: {x_main_data.size(1)}") # get length of sequence dimension
                if x_main_data.size(1) == 0: 
                    continue

                target, new_round, x_main_data, x_prim_weap, x_sec_weap = target.to(self.device), new_round.to(self.device), x_main_data.to(self.device), x_prim_weap.int().to(self.device), x_sec_weap.int().to(self.device)

                # need categorical data as ints for embedding
                x_prim_weap, x_sec_weap = x_prim_weap.int(), x_sec_weap.int()
                # with open('batch_data.txt', 'a') as f:
                #     print(f"Target: {target}, Shape: {target.size()};\n Main_data: {x_main_data}, Shape: {x_main_data.size()};\n Weapon_data: {x_prim_weap}, Shape: {x_prim_weap.size()};\n New_Round: {new_round}, Shape: {new_round.shape};\n\n\n", file = f)
        

                out, self.hidden = self.model(x_main_data, x_prim_weap, x_sec_weap, self.hidden) #hidden is info from past

                out = out.squeeze() # Output comes out of self.model (batch_size, 1) for some reason

                num_preds += len(out)

                loss = self.lossFunc(out, target.float())

                self.optimizer.zero_grad() #zeros grad from previous run
                loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
                self.optimizer.step()#changes weights and bias of parameters based on loss

                # sets prediction depending on whether is above of below 0.5 thresh
                binary_preds = (out > 0.5).float()  # Converts to 0 or 1

                # Compute accuracy
                correct += (binary_preds == target).int().sum()

                accuracy = (binary_preds == target).float().mean() * 100

                # if True in new_round:
                #     hidden = self.model.init_hidden(self.batch_size) # reset hidden state on new round occurance TODO currently doesn reset everyround due to batching, prob need to do in model forward
                # else: 
                self.hidden = (self.hidden[0].detach(), self.hidden[1].detach()) # detach if maintaining to next batch

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy, refresh=True)

                total += self.batch_size # TODO: delete

                metric_dict["loss"] = loss.item()
                metric_dict["acc"] = accuracy

                self.log_scalars("train", metric_dict, batch_idx)

            print(f"Predictions total: {num_preds} guesses: {correct}/{total}")
        


    def test(self, epoch):

        # batches = len(self.testing_data)
        # hidden = self.model.hidden_init() #initialize hidden variable
        self.model.eval()
        loss = 0

        metric_dict = {}
        total = 0
        correct = 0
        num_preds = 0

        with torch.no_grad(): #stops gradient computation - it is unnecessary
            with tqdm(self.test_loader, unit="batch", leave=True) as tepoch:
                for batch_idx, (target, new_round, x_main_data, x_prim_weap, x_sec_weap) in enumerate(tepoch):
                    
                    # print(f"Len of main: {x_main_data.size(1)}") # get length of sequence dimension
                    if x_main_data.size(1) == 0: 
                        continue

                    target, new_round, x_main_data, x_prim_weap, x_sec_weap = target.to(self.device), new_round.to(self.device), x_main_data.to(self.device), x_prim_weap.int().to(self.device), x_sec_weap.int().to(self.device)

                    # need categorical data as ints for embedding
                    x_prim_weap, x_sec_weap = x_prim_weap.int(), x_sec_weap.int()
                    # with open('batch_data.txt', 'a') as f:
                    #     print(f"Target: {target}, Shape: {target.size()};\n Main_data: {x_main_data}, Shape: {x_main_data.size()};\n Weapon_data: {x_prim_weap}, Shape: {x_prim_weap.size()};\n New_Round: {new_round}, Shape: {new_round.shape};\n\n\n", file = f)
            

                    out, self.hidden = self.model(x_main_data, x_prim_weap, x_sec_weap, self.hidden) #hidden is info from past

                    out = out.squeeze() # Output comes out of self.model (batch_size, 1) for some reason

                    num_preds += len(out)

                    loss = self.lossFunc(out, target.float())

                    # self.optimizer.zero_grad() #zeros grad from previous run
                    # loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
                    # self.optimizer.step()#changes weights and bias of parameters based on loss

                    # sets prediction depending on whether is above of below 0.5 thresh
                    binary_preds = (out > 0.5).float()  # Converts to 0 or 1

                    # Compute accuracy
                    correct += (binary_preds == target).int().sum()

                    accuracy = (binary_preds == target).float().mean()

                    # if True in new_round:
                    #     hidden = self.model.init_hidden(self.batch_size) # reset hidden state on new round occurance TODO currently doesn reset everyround due to batching, prob need to do in model forward
                    # else: 
                    self.hidden = (self.hidden[0].detach(), self.hidden[1].detach()) # detach if maintaining to next batch

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy, refresh=True)

                    total += self.batch_size # TODO: delete


                    metric_dict["loss"] = loss.item()
                    metric_dict["acc"] = accuracy

                    self.log_scalars("train", metric_dict, batch_idx)

                print(f"Predictions total: {num_preds} guesses: {correct}/{total}")

        # final_loss = loss/batches
        # print("Final Lost in Test:", final_loss)


    def train_model(self):

        for epoch in range(self.total_epochs):
            self.train(epoch)
            self.test(epoch)


mt = ModelTrainer()

mt.train_model()