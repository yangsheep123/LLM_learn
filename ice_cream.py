# Feature参数
x = [[10,3],[20,3],[25,3],[28,2.5],[30,2],[35,2.5],[40,2.5]]
# Label值
y = [60,85,100,120,140,145,163]
# 参数
w = [0.0,0.0,0.0] # w0,w1,w2
lr=0.0001
interation=10000

for i in range(interation):
    y_pred = [w[0]+w[1]*x[j][0]+w[2]*x[j][1] for j in range(len(y))]
    loss = sum((y_pred[j]-y[j])**2 for j in range(len(y))) /len(y)
    grad_w0 = 2* sum(y_pred[j]-y[j] for j in range(len(y))) /len(y)
    grad_w1 = 2* sum((y_pred[j]-y[j])*x[j][0] for j in range(len(y))) /len(y)
    grad_w2 = 2* sum((y_pred[j]-y[j])*x[j][1] for j in range(len(y))) /len(y)
    w[0] -= grad_w0*lr
    w[1] -= grad_w1*lr
    w[2] -= grad_w2*lr

    if(i%100==0):
        print(loss)

print(f"Final paremeters:w0={w[0]},w1={w[1]},w2={w[2]}")
