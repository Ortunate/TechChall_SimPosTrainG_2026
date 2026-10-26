import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=torch.float32,
    device=device,
)

print("Tensor:", x)
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Device:", x.device)
print("Mean:", x.mean())