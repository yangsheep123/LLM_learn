import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


inputs = torch.tensor([[2,1000],[3,2000],[2,500],[1,800],[4,3000]],dtype=torch.float,device = device)
labels = torch.tensor([[19],[31],[14],[15],[43]],dtype=torch.float,device = device)

# 归一化
mean = inputs.mean(dim=0) #均值
std = inputs.std(dim=0) #标准差
inputs = (inputs-mean) / std
w = torch.ones((2,1),requires_grad=True,device = device)
b = torch.ones((1,),requires_grad = True,device = device)

lr = 0.1
iter = 2000
for i in range(iter):
    loss = torch.mean(torch.square(inputs @ w + b - labels))
    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    w.grad.zero_()
    b.grad.zero_()

    if(i%100==0):
        print(f"loss={loss}")

print(f"训练后的w={w}")
print(f"训练后的b={b}")

