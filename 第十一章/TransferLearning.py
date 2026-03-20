import torchvision.models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
from torchvision import transforms
import torch.nn as nn
import os

def verify_image(image_folder):
    classes = ["Cat","Dog"]
    class_idx = {"Cat":0,"Dog":1}
    samples = []
    for cl in classes:
        # os.path.join(x,y)将x的路径和y的路径拼接起来
        cla_dir = os.path.join(image_folder,cl)
        # os.listdir(cla_dir):排列出这个文件夹的所有文件名
        for fname in os.listdir(cla_dir):
            # 图片必须是这三个后缀名的
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            path = os.path.join(cla_dir,fname)
            try:
                with Image.open(path) as img:
                    # Image的内置函数，检查图片是否完好
                    img.verify()
                samples.append((path,class_idx[cl]))
            except Exception:
                print(f"Warning: Skipping corrupted image {path}")
    return samples

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad=False # 冻结所有层的参数
        for param in self.model.layer4.parameters():
            param.requires_grad = True #解冻第四层的参数
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features,1) # 新生成一个二分类头

    def forward(self,x):
        return torch.sigmoid(self.model(x))
    
class PetData(Dataset):
    def __init__(self,samples,transform = None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        # label属于int型
        path,label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                # transform将图片都变成同一个尺寸,并且img会变成tensor类型
                img = self.transform(img)
        return img,label

def evaluate(model,test_loader):
    model.eval()
    correct = 0
    size_sam = 0
    for inputs,labels in test_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(inputs)
        preds = (outputs>0.5).float()
        correct += (preds==labels).sum().item()
        size_sam += labels.shape[0]
    val_acc = correct / size_sam
    return val_acc

if __name__ == "__main__":
    batch_size = 64
    IMG_SIZE = 128
    EPOCHS = 15
    LR = 0.001
    PRINT_STEP = 100

    samples = verify_image(r"D:/study/训练数据/PetImages")
    # 这两句话是为了每次打乱后都一样，有助于复现
    random.seed(42)
    random.shuffle(samples)

    size = int(len(samples)*0.8)
    train_samples = samples[:size]
    test_samples = samples[size:]

    data_transform  = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = PetData(train_samples,data_transform)
    test_data = PetData(test_samples,data_transform)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True) 
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
    model = CNNModel()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR)
    criterion = nn.BCELoss()

    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        running_loss=0
        # enumerate遍历时返回索引和值
        for step,(inputs,labels) in enumerate(train_loader):
            labels = labels.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % PRINT_STEP == 0:
                avg_loss = running_loss / PRINT_STEP
                print(f"  Step [{step + 1}] - Loss: {avg_loss:.4f}")
                running_loss = 0.0
        val_acc = evaluate(model,test_loader)
        print(f"Validation Accuracy after epoch {epoch + 1}: {val_acc:.4f}")


