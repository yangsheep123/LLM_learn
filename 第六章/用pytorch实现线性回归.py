import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#生成100条数据，其中每个数据有3个feature，这三个feature的取值在0和1之间生成了一个100*3的矩阵
inputs=torch.rand(100,3) 
#生成一个2维的weights3*1矩阵,以便可以和inputs矩阵相乘
weights=torch.tensor([[1.1],[2.2],[3.3]])
#randn生成的数据符合正态分布
bias = torch.tensor(4.4)
#torch.randn()为了拟合实际情况的浮动,生成一个 100 行 1 列的二维张量，张量里的每个元素都是从标准正态分布
targets = inputs @ weights + bias + 0.1*torch.randn(100,1)
inputs=inputs.to(device)
targets=targets.to(device)


w = torch.randn((3,1),requires_grad=True,device=device)
#生成一个 长度为 1 的一维张量，元素是 [0,1) 之间的均匀随机数。
#形状 (1,) 是偏置（bias）的标准写法，在和 (100, 1) 的预测结果相加时，
# PyTorch 会自动广播成 (100, 1)，给每个样本的预测值都加上同一个偏置。
b = torch.rand((1,),requires_grad=True,device=device)
lr=0.003
iteration=10000

for i in range(iteration):
    loss = torch.mean((inputs @ w + b - targets)**2)
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
    # 清零梯度值，否则梯度值会一直累加
    w.grad.zero_()
    b.grad.zero_()

    if(i%100==0):
        print(f"loss={loss}")

print(f"训练后的w={w}")
print(f"训练后的b={b}")
