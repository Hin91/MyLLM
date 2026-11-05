import torch

print("=" * 45)
print("PYTORCH GPU TEST")
print("=" * 45)

print("PyTorch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available!")

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))

properties = torch.cuda.get_device_properties(0)

print("VRAM:", round(properties.total_memory / 1024**3, 2), "GB")
print("BF16 supported:", torch.cuda.is_bf16_supported())

# Создаём матрицы на видеокарте
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

# Выполняем матричное умножение
c = torch.matmul(a, b)

torch.cuda.synchronize()

print("Matrix multiplication:", c.shape)
print("Peak VRAM:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 2
), "GB")

print("GPU TEST PASSED")