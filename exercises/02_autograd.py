import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

w = torch.tensor(1.0, device=device, requires_grad=True)
x = torch.tensor(2.0, device=device)
target = torch.tensor(6.0, device=device)

lr = 0.1

# TODO 1: 计算模型预测值 prediction = w * x
pred = w * x

# TODO 2: 用平方误差计算 loss
loss = (pred - target) ** 2 

# TODO 3: 调用反向传播，计算 w.grad
loss.backward()

print("Before update:", w.item())
print("Loss:", loss.item())
print("Gradient:", w.grad.item())

with torch.no_grad():
    w -= lr * w.grad

print("After update:", w.item())

w.grad.zero_()