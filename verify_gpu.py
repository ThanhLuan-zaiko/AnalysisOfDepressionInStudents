"""
GPU Verification Script for PyTorch.

This script checks if PyTorch can detect and use GPU acceleration.
Run after installing PyTorch to verify GPU support.
"""

import sys


def check_gpu() -> None:
    print("=" * 60)
    print(" PyTorch GPU Verification")
    print("=" * 60)
    print()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print(r"   Run: .\setup_with_gpu.ps1")
        sys.exit(1)

    torch_version = getattr(torch, "__version__", None)
    if not torch_version:
        print("ERROR: Imported 'torch', but it is not a valid PyTorch runtime.")
        print(f"   Module repr: {torch!r}")
        print(f"   Module file: {getattr(torch, '__file__', None)}")
        sys.exit(1)

    print(f"PyTorch version: {torch_version}")
    print()

    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"   GPU {index}: {torch.cuda.get_device_name(index)}")

        try:
            x_tensor = torch.randn(100, 100).cuda()
            y_tensor = torch.randn(100, 100).cuda()
            z_tensor = x_tensor @ y_tensor
            print("GPU computation test: PASSED")
            print(f"   Tensor device: {z_tensor.device}")
        except Exception as exc:
            print(f"GPU computation test: FAILED - {exc}")
    else:
        print("INFO: CUDA is NOT available")
        cpu_tensor = torch.randn(100, 100)
        print("   CPU tensor test: PASSED")
        print(f"   Tensor device: {cpu_tensor.device}")
        print()
        print("   PyTorch is currently using CPU-only mode.")
        print("   To enable GPU, make sure:")
        print("   - NVIDIA GPU: Install CUDA toolkit and cuDNN")
        print("   - AMD GPU: Install ROCm (Linux only)")
        print("   - Intel GPU: Install Intel Extension for PyTorch")

    print()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS (Metal Performance Shaders) is available!")
        try:
            _ = torch.randn(100, 100, device="mps")
            print("MPS tensor test: PASSED")
        except Exception as exc:
            print(f"MPS tensor test: FAILED - {exc}")
    elif hasattr(torch.backends, "mps"):
        print("INFO: Apple MPS is NOT available")

    print()
    print("=" * 60)

    if torch.cuda.is_available():
        print("GPU acceleration is READY!")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple MPS acceleration is READY!")
    else:
        print("INFO: Running in CPU mode. Performance may be slower for large datasets.")

    print("=" * 60)
    print()


if __name__ == "__main__":
    check_gpu()
