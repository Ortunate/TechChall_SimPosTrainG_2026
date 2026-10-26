import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

# 目标关系：y = 3x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y = 3 * x + 1

model = nn.Linear(1, 1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

for step in range(100):
    optimizer.zero_grad()

    # TODO 1: 调用 model(x) 得到预测值
    pred = model(x)

    # TODO 2: 使用 nn.functional.mse_loss() 计算损失
    loss = nn.functional.mse_loss(pred, y)

    # TODO 3: 反向传播并更新参数
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step={step}, loss={loss.item():.6f}")

print("weight:", model.weight.item())
print("bias:", model.bias.item())