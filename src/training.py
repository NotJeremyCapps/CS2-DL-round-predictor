
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
from torchsummary import summary


from dataset import CS2PredictionDataset, split_dataset

from model import CS2LSTM

# torch.set_printoptions(profile="full")

DEMO_PATH = "../../demos/"


class ModelTrainer():

    def __init__(self):

        self.total_epochs = 120

        self.batch_size = 64

        self.seq_len = 120

        self.n_hidden = 256
        self.n_layers = 3

        self.batch_interval = 3 # Detach every 3 batches

        self.model = CS2LSTM(n_feature=None, out_feature=1,n_hidden=self.n_hidden,n_layers=self.n_layers)

        self.hidden = self.model.init_hidden(self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dir = os.path.join('logs', datetime.now().strftime('%B%d_%H_%M_%S'))

        try:
            os.mkdir(self.log_dir)
            print(f"Log Directory '{self.log_dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{self.log_dir}' already exists.")
        except Exception as e:
            print(f"An error occurred: {e}")

        writer_pth = os.path.join(self.log_dir, "tb_logs")
        self.writer = SummaryWriter(writer_pth)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)  # Wrap model for multi-GPU

        # hidden = self.model.hidden_init() #initialize hidden variable

        self.model.to(self.device)  # Move model to GPUs

        print(self.model)

        # self.lossFunc = nn.L1Loss()
        self.lossFunc = nn.BCEWithLogitsLoss()

        print(self.lossFunc)

        with open(os.path.join(self.log_dir,"config.txt"), "a") as f:
                        f.write(f"{self.model}\n\n")
                        f.write(f"{self.seq_len}\n")
                        f.write(f"{self.n_hidden}\n")
                        f.write(f"{self.n_layers}\n\n")
                        f.write(f"{self.lossFunc}\n\n")
                        f.write(f"{self.batch_size}\n")
                        

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.split_dataset = False

        if self.split_dataset:
            split_dataset(0.8, os.path.join(DEMO_PATH, "preprocessed"))

        self.trainset = CS2PredictionDataset(list_path=os.path.join(DEMO_PATH, "preprocessed/rounds_train.txt"), sequence_length=self.seq_len)
        self.testset = CS2PredictionDataset(list_path=os.path.join(DEMO_PATH, "preprocessed/rounds_test.txt"), sequence_length=self.seq_len)

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


    def log_scalars(self, global_tag, metric_dict, global_step):

        for tag, value in metric_dict.items():
            self.writer.add_scalar(f"{global_tag}/{tag}", value, global_step)

    def train(self, epoch, prof=None): #loss function and optimizer should be prebuilt functions

        self.model.train() #inherited function from model.py, sets model in training modes

        metric_dict = {}
        total = 0
        correct = 0
        num_preds = 0
        running_acc = 0
        running_loss = 0

        plyr11_inst = 0

        # hidden = None
        with tqdm(self.train_loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (target, new_round, x_main_data, x_prim_weap, x_sec_weap) in enumerate(tepoch):
                
                # print(f"Len of main: {x_main_data.size(1)}") # get length of sequence dimension
                if x_main_data.size(1) == 0: 
                    continue
                elif x_main_data.size(1) > 146: 
                    plyr11_inst += 1
                    print(f"Player 11 found. Num Instances: {plyr11_inst}")
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.zero_grad() #zeros grad from previous run
                loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
                self.optimizer.step()#changes weights and bias of parameters based on loss

                # with open(os.path.join(self.log_dir,"preds.txt"), "a") as f:
                #     f.write(f"Output:{out}\n\n")

                # Convert out to probabilities
                out = torch.sigmoid(out)

                # sets prediction depending on whether is above of below 0.5 thresh
                binary_preds = (out > 0.5).int()  # Converts to 0 or 1

                # with open(os.path.join(self.log_dir,"preds.txt"), "a") as f:
                #     f.write(f"Binary Targets:{target}\n")
                #     f.write(f"Binary Preds:{binary_preds}\n\n")


                # Compute accuracy
                correct += (binary_preds == target).int().sum()

                accuracy = ((binary_preds == target).int().sum() / len(target)) * 100

                running_acc += accuracy.item()
                running_loss += loss.item()

                if True in new_round: # Detach if new round or batch interval met
                    self.hidden = self.model.init_hidden(self.batch_size) # reset hidden state on new match occurance TODO 
                else: 
                    self.hidden = (self.hidden[0].detach(), self.hidden[1].detach()) # detach if maintaining

                # with open(os.path.join(self.log_dir,"new_round.txt"), "a") as f:
                #         f.write(f"[Batch {batch_idx}] New Round:{new_round}\n\n")


                total += self.batch_size # TODO: delete

                # prof.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item(), refresh=True)

                # metric_dict["batch_loss"] = 
                # metric_dict["batch_acc"] = running_acc / total

                # self.log_scalars("train", metric_dict, epoch)


            metric_dict["loss"] = running_loss / len(self.train_loader) # Divides by number of batches
            metric_dict["acc"] = correct / total

            self.log_scalars("train", metric_dict, epoch)

            print(f"[TRAIN] [EPOCH {epoch}] - Predictions total: {num_preds} guesses: {correct}/{total} | {(correct/total) * 100}%")
        


    def test(self, epoch, prof=None):

        # batches = len(self.testing_data)
        # hidden = self.model.hidden_init() #initialize hidden variable
        self.model.eval()
        loss = 0

        metric_dict = {}
        total = 0
        correct = 0
        num_preds = 0

        running_loss = 0

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

                    # with open(os.path.join(self.log_dir,"test_raw_preds.txt"), "a") as f:
                    #     f.write(f"Output:{out}\n\n")

                    # self.optimizer.zero_grad() #zeros grad from previous run
                    # loss.backward() #uses chain rule to calculate gradient of loss (rate of change of loss)
                    # self.optimizer.step()#changes weights and bias of parameters based on loss

                    # Convert out to probabilities
                    out = torch.sigmoid(out)

                    # with open(os.path.join(self.log_dir,"test_prob_preds.txt"), "a") as f:
                    #     f.write(f"Sigmoid Output:{out}\n\n")

                    # sets prediction depending on whether is above of below 0.5 thresh
                    binary_preds = (out > 0.5).int()  # Converts to 0 or 1

                    with open(os.path.join(self.log_dir,"test_preds.txt"), "a") as f:
                        f.write(f"Binary Targets:{target}\n")
                        f.write(f"Prob Preds:{out}\n\n")

                    # Compute accuracy
                    correct += (binary_preds == target).int().sum()

                    accuracy = (binary_preds == target).float().mean()

                    running_loss += loss.item()

                    if True in new_round: # Detach if new round or batch interval met
                        self.hidden = self.model.init_hidden(self.batch_size) # reset hidden state on new match occurance TODO 
                    else: 
                        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach()) # detach if maintaining

                    total += self.batch_size # TODO: delete

                    # prof.step()

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy, refresh=True)



                metric_dict["loss"] = running_loss / len(self.test_loader) # Divides by number of batches
                metric_dict["acc"] = correct / total

                self.log_scalars("test", metric_dict, epoch)


                print(f"[TEST] [EPOCH {epoch}] - Predictions total: {num_preds} guesses: {correct}/{total} | {(correct/total) * 100}%")

        # final_loss = loss/batches
        # print("Final Lost in Test:", final_loss)

    # def train_model(self):
    #         with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),  
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.log_dir,'profiler')),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    #         ) as prof:
    #             for epoch in range(self.total_epochs):
    #             #for step, batch_data in enumerate(train_loader):
    #                 print("Step is being executed")
    #                 prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
    #                 #if step >= 1 + 1 + 3:
    #                 #    break
    #                 self.train(epoch, prof)
    #                 self.test(epoch, prof)



    def train_model(self):

        for epoch in range(self.total_epochs):
            self.train(epoch)
            self.test(epoch)

            epoch_mdl_pth = os.path.join(self.log_dir, f"epoch_{epoch}.pt")
            torch.save(self.model, epoch_mdl_pth)


mt = ModelTrainer()

mt.train_model()
