#!/usr/bin/env python
"""
GPU Installation Verification Script
Run this on your RTX 4080 machine to verify everything is working.
"""

import torch
import sys

def test_cuda_installation():
    """Test CUDA and PyTorch GPU installation."""
    print("=" * 60)
    print("🔧 GPU Installation Verification")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available! Check installation.")
        print("\nTroubleshooting steps:")
        print("1. Install NVIDIA drivers (522.06+)")
        print("2. Install CUDA toolkit")
        print("3. Reinstall PyTorch with GPU support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # Check GPU details
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    
    if gpu_count == 0:
        print("❌ No GPUs detected!")
        return False
    
    # Check each GPU
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    print(f"\nCUDA version: {cuda_version}")
    
    # Check cuDNN
    cudnn_available = torch.backends.cudnn.enabled
    print(f"cuDNN available: {cudnn_available}")
    if cudnn_available:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Test basic GPU operations
    print("\n" + "=" * 40)
    print("Testing GPU Operations")
    print("=" * 40)
    
    try:
        # Test tensor creation on GPU
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        print(f"✅ Tensor creation on GPU: {x.device}")
        
        # Test computation
        y = torch.matmul(x, x.T)
        print(f"✅ Matrix multiplication on GPU: {y.shape}")
        
        # Test memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1e6
        memory_reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"✅ GPU memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
        
        # Test mixed precision
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        with autocast():
            z = torch.matmul(x, y[:1000, :1000])
        print("✅ Mixed precision (autocast) working")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL GPU TESTS PASSED!")
    print("🚀 Ready for high-performance training!")
    print("=" * 60)
    
    # Performance estimate
    print(f"\nExpected performance with RTX 4080:")
    print(f"• Training speed: ~8x faster than CPU")
    print(f"• Batch size 8 with 128³ images: Should work smoothly")
    print(f"• Mixed precision: ~30% speed boost")
    print(f"• Memory usage: ~8-12 GB for ensemble training")
    
    return True

def check_requirements():
    """Check if all required packages are installed."""
    print("\n" + "=" * 40)
    print("Checking Required Packages")
    print("=" * 40)
    
    requirements = [
        'torch', 'torchvision', 'monai', 'numpy', 'scipy',
        'scikit-learn', 'tqdm', 'yaml', 'nibabel'
    ]
    
    missing = []
    for package in requirements:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All required packages installed")
    return True

if __name__ == "__main__":
    gpu_ok = test_cuda_installation()
    packages_ok = check_requirements()
    
    if gpu_ok and packages_ok:
        print("\n🎉 SYSTEM READY FOR GPU TRAINING! 🎉")
    else:
        print("\n❌ Please fix the issues above before training")
