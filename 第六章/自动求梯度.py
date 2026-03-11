import torch
#True表示需要求梯度，1.0表示这是一个标量（0 维张量），数值就是 1.0（浮点数）
x=torch.tensor(1.0,requires_grad=True) 
y=torch.tensor(1.0,requires_grad=True)

v=3*x+4*y
u=v**2
z=torch.log(u)
z.backward() #反向传播求梯度

print(x.grad)
print(y.grad)


