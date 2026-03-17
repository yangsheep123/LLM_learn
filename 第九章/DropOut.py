import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class MyModel(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,10)
        )