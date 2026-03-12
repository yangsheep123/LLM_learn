import torch
from torch.utils.data import Dataset
import pandas as pd

class TitanicData(Dataset): # 这里的括号表示这个类继承Dataset类
    def __init__(self,file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.feature_size = len(self.data.columns) - 1

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
            df[base_features[i]] = (df[base_features[i]] - self.mean[i]) / self.std[i]
        return df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x = self.data.drop(columns=["Survived"]).iloc[idx].values
        y = self.data["Survived"].iloc[idx]
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    
from torch.utils.data import DataLoader

Dataset = TitanicData(r"D:/study/大模型入门/第七章/titanic/train.csv")
# shuffle=True:打乱数据
dataloader = DataLoader(Dataset,batch_size = 32,shuffle=True)

for i,j in dataloader:
    print(i.shape,j.shape)
    break

    