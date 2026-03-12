import torch
import torch.nn as nn

class Example(nn.model):
    def __init__(self,dim):
        super.__init__() # 先调用父类的init函数
        self.linear = torch.Linear(dim,1)

    def forward(self,x):
        return torch.sigmoid(self.linear(x))