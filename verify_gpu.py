"""
GPU Verification Script for PyTorch
This script checks if PyTorch can detect and use GPU acceleration.
Run after installing PyTorch to verify GPU support.
"""

import sys

def check_gpu():
    print("=" * 60)
    print(" PyTorch GPU Verification")
    print("=" * 60)
    print()
    
    try:
        import torch
    except ImportError:
        print("❌ ERROR: PyTorch is not installed!")
        print("   Run: uv add torch")
        sys.exit(1)
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print()
    
    # Check CUDA (NVIDIA)
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test tensor on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = x @ y
            print(f"✅ GPU computation test: PASSED")
            print(f"   Tensor device: {z.device}")
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
    else:
        print("ℹ️  CUDA is NOT available")
        
        # Check if running on CPU
        cpu_tensor = torch.randn(100, 100)
        print(f"   CPU tensor test: PASSED")
        print(f"   Tensor device: {cpu_tensor.device}")
        print()
        print("   This means PyTorch is using CPU-only mode.")
        print("   To enable GPU, make sure:")
        print("   - NVIDIA GPU: Install CUDA toolkit and cuDNN")
        print("   - AMD GPU: Install ROCm (Linux only)")
        print("   - Intel GPU: Install Intel Extension for PyTorch")
    
    print()
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Apple MPS (Metal Performance Shaders) is available!")
        try:
            x = torch.randn(100, 100, device='mps')
            print(f"✅ MPS tensor test: PASSED")
        except Exception as e:
            print(f"❌ MPS tensor test: FAILED - {e}")
    elif hasattr(torch.backends, 'mps'):
        print("ℹ️  Apple MPS is NOT available")
    
    print()
    print("=" * 60)
    
    # Summary
    if torch.cuda.is_available():
        print("✅ GPU acceleration is READY!")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Apple MPS acceleration is READY!")
    else:
        print("ℹ️  Running on CPU mode. Performance may be slower for large datasets.")
    
    print("=" * 60)
    print()

if __name__ == "__main__":
    check_gpu()
