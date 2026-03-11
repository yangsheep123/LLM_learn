import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs=torch.rand(100,3)
weights=torch.tensor([[1.1],[2.2],[3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1 * torch.randn(100,1)
inputs = inputs.to(device)
targets = targets.to(device)

writer = SummaryWriter(log_dir='D:/study/')

w = torch.rand((3,1),requires_grad=True,device=device)
b = torch.rand((1,),requires_grad=True,device=device)
lr=0.003
iter=10000

for i in range(iter):
    loss = torch.mean(torch.square(inputs @ w + b - targets))
    loss.backward()
    # 记录loss值，三个参数：tag，loss值，第几步
    writer.add_scalar('loss/train',loss.item(),i)
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    w.grad.zero_()
    b.grad.zero_()

    if(i%100==0):
        print(f"loss={loss}")

print(f"训练后的w={w}")
print(f"训练后的b={b}")
