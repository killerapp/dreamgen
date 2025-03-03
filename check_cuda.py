import torch
import platform

print(f'PyTorch version: {torch.__version__}')

# Check for CUDA (NVIDIA GPUs)
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')

# Check for MPS (Apple Silicon)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f'MPS available: {mps_available}')
if mps_available:
    print(f'Apple Silicon: {platform.processor()}')
    
# Summary
if torch.cuda.is_available():
    print("\nSystem has NVIDIA GPU with CUDA support")
elif mps_available:
    print("\nSystem has Apple Silicon with MPS support")
else:
    print("\nNo GPU acceleration available (CPU only)")
