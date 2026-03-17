import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class NumData(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        self.images,self.labels = self.load_data()

    def load_data(self):
        images = []
        labels = []
        with open(self.file_path,"r") as f:
            next(f)
            for line in f:
                line = line.strip().split(",")
                images.append([float(x) for x in line[1:]])
                labels.append(int(line[0]))
        return images,labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image)
        image = image/255.0
        image = (image - 0.1307) / 0.3081 # 标准化
        return image,torch.tensor(label)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential会顺序链接内部各个模块
        # 调用 self.model(x) 时，数据会依次通过容器内的每一层，实现前向传播，计算出输出层的logits
        self.model = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.model(x)
    
batch_size=64
train_data = NumData(r"D:/study/大模型入门/第八章/mnist/mnist_train.csv")
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = NumData(r"D:/study/大模型入门/第八章/mnist/mnist_test.csv")
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
lr = 0.1
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(),lr)
epochs = 10
# nn.CrossEntropyLoss 是一个类,因此要先创建这个类的对象，再传入参数
# 并且这个类自动将输出层的logits使用software函数，转换为最终的输出值
criterion = nn.CrossEntropyLoss()

#训练
model.train()
for epoch in range(epochs):
    total_loss = 0
    total =0 
    correct = 0
    for images,labels in train_loader:
        # 返回一个形状为[批量大小，10]形状的tensor,
        outputs = model(images)
        loss = criterion(outputs,labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += labels.shape[0]
        preds = torch.argmax(outputs,dim = 1)
        correct += (preds==labels).sum().item()
    avg_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

model.eval()    
correct = 0
total = 0
with torch.no_grad():    
    for images,labels in test_loader:
        outputs = model(images)    
        preds = torch.argmax(outputs,dim = 1)
        correct += (preds==labels).sum().item()  
        total += labels.shape[0]
        train_acc = 100 * correct / total
    print(f"Test Accuracy: {train_acc:.2f}%")

