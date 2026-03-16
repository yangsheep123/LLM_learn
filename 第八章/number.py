import torch
from torch.utils.data import DataLoader,Dataset

class NumData(Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        self.images,self.labels = self.load_data()


    def load_data(self):
        images = []
        labels = []
        with open(self.file_path,"r") as f:
            next(f) # 跳过标题行
            for line in f:
                line = line.strip() #去掉换行符
                item = line.split(",") # 根据逗号划分每个字符串
                # line[1:]得到的是一个列表
                # images.append([line[1:]]) #images是三维
                # x是字符串
                images.append([float(x) for x in item[1:]])
                labels.append(int(item[0])) #labels是一维
        return images,labels

    def __len__(self):
        return len(self.labels)
    
    # 对image进行归一化和标准化
    def __getitem__(self,idx):
        image = self.images[idx]
        image = torch.tensor(image)
        label = self.labels[idx]
        image = image/255.0
        # image = (image - image.mean()) / image.std()
        # 0.1307和0.3081是全部图片的均值和方差，这样设置有利于学习到全局的规律
        image = (image - 0.1307) / 0.3081 # 标准化
        return image,torch.tensor(label)
    
def relu(x):
    # 将输入张量中小于 min 的值设置为 min
    # torch.clamp(input, min=None, max=None, *, out=None) → Tensor
    return torch.clamp(x,min=0)

def relu_grad(x):
    # 如果x>0，则(x>0)为TRUE，再将其转换为float型数据
    return (x>0).float()

def softmax(x):
    # 减去最大值防止数据超过float的范围
    x_exp = torch.exp(x - x.max(dim=1,keepdim = True).values)
    return (x_exp / x_exp.sum(dim=1,keepdim=True))

def cross_entropy(pred,labels):
    # pred是输出层的激活值，是样本数*类别数大小的矩阵
    # shape[0]:取第一维的大小
    N = pred.shape[0]
    # one_hot形状与pred相同，元素为全0
    one_hot = torch.zeros_like(pred)
    # one_hot的第一个索引和第二个索引都是tensor类型，torch会进行按位配对
    one_hot[torch.arange(N),labels] = 1
    # 防止对0取对数
    loss = -(one_hot * torch.log(pred + 1e-8)).sum() / N
    # loss依旧是N*10的tensor类型
    return loss,one_hot


batch_size=64
train_data = NumData(r"D:/study/大模型入门/第八章/mnist/mnist_train.csv")
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = NumData(r"D:/study/大模型入门/第八章/mnist/mnist_test.csv")
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

# 初始化weights和bias
# weights是一个一维列表，每个元素都是一个tensor
# weights[0]表示第一个隐藏层的权重系数
weights = []
bias = []
layer_sizes = [28*28, 128, 128, 128, 64, 10]
# layer_sizes[:-1]表示每一层的输入变量
# layer_sizes[1:]表示每一层的输出变量
for in_size,out_size in zip(layer_sizes[:-1],layer_sizes[1:]):
    # w的方差是torch.sqrt(torch.tensor(2/in_size))，这是为了避免梯度爆炸
    w = torch.randn(in_size,out_size) * torch.sqrt(torch.tensor(2/in_size))
    # 后面加bias的时候，会自动广播
    b = torch.zeros(out_size)
    weights.append(w)
    bias.append(b)

epoch = 10
lr = 0.1

for e in range(epoch):
    total_loss = 0
    for x,label in train_loader:
        activation = [x]
        logits = []
        N = x.shape[0]

        # 前向传播
        for w,b in zip(weights[:-1],bias[:-1]):
            z = activation[-1] @ w + b
            logits.append(z)
            a = relu(z)
            activation.append(a)
        # 输出层
        z = activation[-1] @ weights[-1] + bias[-1]
        y_pred = softmax(z)
        logits.append(z)
        # activation.append(y_pred)

        # 损失
        loss,one_hot = cross_entropy(y_pred,label)
        total_loss += loss.item()

        # 反向传播
        w_grads = [None] * len(weights)
        b_grads = [None] * len(bias)
        dL_dz = (y_pred - one_hot) / N
        w_grads[-1] = activation[-1].t() @ dL_dz
        b_grads[-1] = dL_dz.sum(dim=0)
        for i in range(len(weights)-2,-1,-1):
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(logits[i])
            w_grads[i] = activation[i].t() @ dL_dz
            b_grads[i] = dL_dz.sum(dim=0)

        with torch.no_grad():
            # 为什么要这样子一个个算？因为weights和w_grads都是列表，列表不支持直接减
            for i in range(len(weights)):
                weights[i] -= w_grads[i]*lr
                bias[i] -= b_grads[i]*lr
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {e+1}/{epoch}, Loss: {avg_loss:.4f}")

# 测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.view(-1, layer_sizes[0])
        y = labels
        a = x
        for W, b in zip(weights[:-1], bias[:-1]):
            a = relu(a @ W + b)
        logits = a @ weights[-1] + bias[-1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Test Accuracy: {correct/total*100:.2f}%")
