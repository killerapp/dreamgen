import sys
import torch
import platform
import os

def check_cuda_details():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"\nCUDA Environment Variables:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("\nPyTorch build info:")
        print(torch.__config__.show())
        
        print("\nCUDA not available. Possible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers are not properly installed")
        print("3. CUDA toolkit is not properly installed")
        print("4. GPU is not CUDA-capable")

if __name__ == "__main__":
    check_cuda_details()
