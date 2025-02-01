import torch

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
