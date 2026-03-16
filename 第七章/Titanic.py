import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader,Dataset

class MyModule(nn.Module):
    def __init__(self,dim):
        super().__init__()
        # 调用torch的线性层，第一个参数是输入特征的个数，第二个参数是输出的个数
        self.linear = nn.Linear(dim,1)
        

    def forward(self,x):
        # 要调用forward函数，使用module(inputs)
        return torch.sigmoid(self.linear(x))

    # def forward(self, x):
    #     return self.linear(x)  # 不加 sigmoid
    
class MyData(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        self.dataset = self.load_data()
        self.feature_size = len(self.dataset.columns) - 1


    def load_data(self):
        # df 是 Pandas 库中的 DataFrame 数据类型
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df,columns=["Sex", "Embarked"], dtype = int)
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        self.mean = df[base_features].mean()
        self.std = df[base_features].std()
        #标准化数据
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        # .values 的作用是将 pandas 的 Series/DataFrame 转换为 numpy 数组，
        # 适配后续的张量转换、数值计算等操作。
        x = self.dataset.drop(columns=["Survived"]).iloc[idx].values
        y = self.dataset["Survived"].iloc[idx]
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    
# 定义数据和模型
train_data = MyData(r"D:\study\大模型入门\第七章\titanic\train.csv")
val_data = MyData(r"D:\study\大模型入门\第七章\titanic\validation.csv")
module = MyModule(train_data.feature_size)
# 将module转为训练模式，因为很多模型训练模式和验证模式不一样
module.train()

# SGD表示随机梯度下降，传入模型的可训练参数和学习率
optimizer = torch.optim.SGD(module.parameters(),lr=0.1)
epoch = 100
correct = 0
# 一个epoch里有多个batch,
# epoch 1
#   ├── batch 1 (32条数据)
#   ├── batch 2 (32条数据)
#   ├── batch 3 (32条数据)
#   └── ... (假设训练集714条，共 714/32 ≈ 23个batch)

for i in range(epoch):
    total_loss = 0
    correct = 0
    for feature,label in DataLoader(train_data,batch_size = 256,shuffle = True):
        optimizer.zero_grad()
        outputs = module(feature).squeeze()
        correct += torch.sum(((outputs>=0.5)==label))
        # correct += torch.sum(((torch.sigmoid(outputs) >= 0.5) == label))
        loss = torch.nn.functional.binary_cross_entropy(outputs,label)
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, label)
        total_loss += loss.item()
        loss.backward()
        # 优化器对训练参数进行梯度更新
        optimizer.step()
    print(f"total_loss={total_loss},迭代次数:{i}")
    print(f"正确率为：{correct / len(train_data)}")

# 转成验证模式
module.eval()
with torch.no_grad():
    for features,labels in DataLoader(val_data,batch_size = 256,shuffle = True):
        correct = 0
        outputs = module(features).squeeze()
        correct += torch.sum((outputs>=0.5) == labels)

print(f"验证集的正确率：{correct / len(val_data)}")