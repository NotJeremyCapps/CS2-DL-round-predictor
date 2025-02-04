from dataset import CS2PredictionDataset
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as P
from torch.utils.data import Subset
import pandas as pd
from torch.utils.data import DataLoader
from training import ModelTrainer

data = CS2PredictionDataset('../game_demos/preprocessed/de_anubis/rounds.txt', 30)
loaded_data = DataLoader(data, batch_size = 128,shuffle = False)
torch.set_printoptions(threshold=torch.inf)  
i = 0   
'''
for batch in loaded_data:
    with open('loadeddata' + str(i) + '.txt', 'w') as f:
        print(batch, file = f)
        i+=1
'''

for batch in loaded_data:
    train = ModelTrainer(len(batch[0]), 128,4,1,batch) #128 data points placed in (batch size of 128)
    train.train_model()
