import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class MyModel(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(28*28,128,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),


            nn.Linear(128,128,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,128,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,64,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,10)
        )