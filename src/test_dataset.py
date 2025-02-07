from dataset import CS2PredictionDataset
from training import ModelTrainer
import torch

test = CS2PredictionDataset('../game_demos/preprocessed/de_anubis/rounds.txt', 30)
#test.__getitem__(0)
test.__len__()

test2 = ModelTrainer()