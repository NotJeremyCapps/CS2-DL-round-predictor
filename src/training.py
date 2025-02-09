
#have to initialize hidden state of model in model

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import math


from dataset import CS2PredictionDataset

from model import CS2LSTM

# torch.set_printoptions(profile="full")


class ModelTrainer():

    def __init__(self):

        self.total_epochs = 20 

        self.batch_size = 28

        self.seq_len = 60

        self.model = CS2LSTM(n_feature=None, out_feature=1,n_hidden=self.seq_len,n_layers=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.devices = []

        for i in range(torch.cuda.device_count()):
            self.devices.append("cuda:" + str(i))

        print("devices!: ",self.devices)


        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Wrap model for multi-GPU

        # hidden = self.model.hidden_init() #initialize hidden variable

        self.model.to(self.devices[0])  # Move model to GPUs
        #print("MODELDEV: ", self.model.get_device())


        self.lossFunc = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizers = []

        for i in range(len(self.devices)):
            self.optimizers.append(optim.Adam(self.model.parameters(), lr=0.001))

        self.trainset = CS2PredictionDataset(list="../game_demos/preprocessed/de_anubis/rounds.txt", sequence_length=self.seq_len)

        # self.training_data, self.testing_data = torch.utils.data.random_split(data_set, [0.75, 0.25])

        self.train_loader = DataLoader(dataset=self.trainset,
                                     batch_size=self.batch_size,
                                     num_workers=0,
                                     drop_last=True)

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
        #loss = 0
        binary_preds = []
        hidden = None
    
        with tqdm(self.train_loader, unit="batch", leave=True) as tepoch:
            for batch, (target, new_round, x_main_data, x_prim_weap, x_sec_weap) in enumerate(tepoch):

                #for i in range(torch.cuda.device_count()):
                #target, new_round,x_main_data, x_prim_weap, x_sec_weap = target.to(self.device), new_round.to(self.device), x_main_data.to(self.device), x_prim_weap.int().to(self.device), x_sec_weap.int().to(self.device)

                # need categorical data as ints for embedding
                x_prim_weap, x_sec_weap = x_prim_weap.int(), x_sec_weap.int()
                # with open('passedtotraining_seq3.txt', 'a') as f:
                #     print(f"Target: {target}, Shape: {target.size()};\n Main_data: {x_main_data}, Shape: {x_main_data.size()};\n Weapon_data: {x_prim_weap}, Shape: {x_prim_weap.size()};\n New_Round: {new_round}, Shape: {new_round.shape};\n\n\n", file = f)


                #out = []

                #hidden = []


                targets = torch.split(target, math.ceil(self.batch_size/torch.cuda.device_count()))
                new_rounds = torch.split(new_round, math.ceil(self.batch_size/torch.cuda.device_count()))
                x_main_datas = torch.split(x_main_data, math.ceil(self.batch_size/torch.cuda.device_count()))
                x_prim_weaps = torch.split(x_prim_weap, math.ceil(self.batch_size/torch.cuda.device_count()))
                x_sec_weaps = torch.split(x_sec_weap, math.ceil(self.batch_size/torch.cuda.device_count()))
                
                targets_list = []
                new_rounds_list = []
                x_main_datas_list = []
                x_prim_weaps_list = []
                x_sec_weaps_list = []


                for i in range(len(self.devices)):
                    targets_list.append(targets[i].to(self.devices[i]))
                    new_rounds_list.append(new_rounds[i].to(self.devices[i]))
                    x_main_datas_list.append(x_main_datas[i].to(self.devices[i]))
                    x_prim_weaps_list.append(x_prim_weaps[i].to(self.devices[i]))
                    x_sec_weaps_list.append(x_sec_weaps[i].to(self.devices[i]))
                    print("i: ", i)

                print("BEFORE")

                for i in range(len(self.devices)):
                    print(self.devices[i])

                    print("targets")
                    #out.append(None)
                    #hidden.append(None)

                    out, hidden = self.model(x_main_datas_list[i], x_prim_weaps_list[i], x_sec_weaps_list[i], hidden) #hidden is info from past

                    out = out.squeeze() # Output comes out of self.model (batch_size, 1) for some reason

                    num_preds += len(out)

                    loss = 0

                    loss += self.lossFunc(out, targets_list[i].float())

                    self.optimizer.zero_grad() #zeros grad from previous run
                    loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
                    self.optimizers[i].step()#changes weights and bias of parameters based on loss

                    # sets prediction depending on whether is above of below 0.5 thresh
                    binary_preds.append((out > 0.5).float())  # Converts to 0 or 1

                # Compute accuracy
                    correct += (binary_preds[i] == targets_list[i]).int().sum()

                    accuracy = (binary_preds[i] == targets_list[i]).float().mean()

                    if True in new_rounds_list[i]:
                        hidden = self.model.module.init_hidden(self.batch_size) # reset hidden state on new round occurance TODO currently doesn reset everyround due to batching, prob need to do in model forward
                    else: 
                        hidden = (hidden[0].detach(), hidden[1].detach()) # detach if maintaining to next batch

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy, refresh=True)

                    total += self.batch_size # TODO: delete

                    metric_dict["loss"] = loss.item()
                    metric_dict["acc"] = accuracy

                    self.log_scalars("train", metric_dict, batch)

            print(f"Predictions total: {num_preds} guesses: {correct}/{total}")
        


    def test(self, epoch):

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

        for epoch in range(self.total_epochs):
            self.train(epoch)



mt = ModelTrainer()

mt.train_model()