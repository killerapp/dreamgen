import sys
import torch
import platform
import os

def check_torch_details():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check for CUDA (NVIDIA GPUs)
    print(f"\n=== NVIDIA CUDA Support ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA Environment Variables:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Check for MPS (Apple Silicon)
    print(f"\n=== Apple Silicon MPS Support ===")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(f"MPS available: {mps_available}")
    print(f"MPS built: {mps_built}")
    
    if not mps_available and mps_built:
        print("MPS is built but not available. This might be because:")
        print("1. You're not running on Apple Silicon hardware")
        print("2. The MPS backend is disabled")
    
    # Show PyTorch build info
    print("\n=== PyTorch Build Information ===")
    print(torch.__config__.show())
    
    # Summary and troubleshooting
    print("\n=== Summary ===")
    if torch.cuda.is_available():
        print("✅ NVIDIA GPU with CUDA support is available")
    elif mps_available:
        print("✅ Apple Silicon with MPS support is available")
    else:
        print("❌ No GPU acceleration available (CPU only)")
        print("\nPossible reasons for no GPU acceleration:")
        print("1. PyTorch was installed without GPU support")
        print("2. GPU drivers are not properly installed")
        print("3. Hardware doesn't support GPU acceleration")
        print("4. Environment configuration issues")

if __name__ == "__main__":
    check_torch_details()
