#!/usr/bin/env python
"""
Verification script to ensure all code changes are ready for GPU training.
Does not require GPU to be present - just validates the configuration and code.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from pathlib import Path

def verify_config():
    """Verify the configuration file has all GPU optimizations."""
    print("🔧 Verifying Configuration...")
    
    config_path = "configs/training/ensemble_config_reordered.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check training parameters
    training = config['global']['training']
    assert training['batch_size'] == 8, f"Batch size should be 8, got {training['batch_size']}"
    assert training['device'] == 'cuda', f"Device should be cuda, got {training['device']}"
    assert training['precision'] == '16-mixed', f"Precision should be 16-mixed, got {training['precision']}"
    assert training['num_workers'] == 8, f"Num workers should be 8, got {training['num_workers']}"
    
    # Check data configuration
    data = config['data']
    assert data['num_workers'] == 8, f"Data workers should be 8, got {data['num_workers']}"
    assert data['pin_memory'] == True, f"Pin memory should be True, got {data['pin_memory']}"
    assert data['cache_rate'] == 0.3, f"Cache rate should be 0.3, got {data['cache_rate']}"
    
    # Check hardware configuration
    hardware = config['hardware']
    assert hardware['device'] == 'cuda', f"Hardware device should be cuda, got {hardware['device']}"
    assert hardware['gpus'] == 1, f"GPUs should be 1, got {hardware['gpus']}"
    
    # Check model image sizes
    for model_name, model_config in config['models'].items():
        if 'img_size' in model_config['params']:
            img_size = model_config['params']['img_size']
            assert img_size == [128, 128, 128], f"{model_name} img_size should be [128,128,128], got {img_size}"
    
    print("✅ Configuration verified - all GPU optimizations present")
    return True

def verify_dataset_size():
    """Verify dataset.py has correct image size."""
    print("🔧 Verifying Dataset Image Size...")
    
    dataset_path = "axon_ia/data/dataset.py"
    with open(dataset_path, 'r') as f:
        content = f.read()
    
    # Check for 128x128x128 target size
    if "target_size = (128, 128, 128)" in content:
        print("✅ Dataset image size updated to 128³")
    else:
        print("❌ Dataset still has old image size - needs manual update")
        return False
    
    return True

def verify_mixed_precision():
    """Verify mixed precision imports are present."""
    print("🔧 Verifying Mixed Precision Support...")
    
    trainer_path = "scripts/train_ensemble_enhanced.py"
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for mixed precision imports
    if "from torch.cuda.amp import GradScaler, autocast" in content:
        print("✅ Mixed precision imports added")
    else:
        print("❌ Missing mixed precision imports")
        return False
    
    # Check for autocast usage
    if "with autocast():" in content:
        print("✅ Mixed precision training loop updated")
    else:
        print("❌ Missing autocast in training loop")
        return False
    
    return True

def verify_model_parameters():
    """Verify model parameters are compatible."""
    print("🔧 Verifying Model Parameters...")
    
    config_path = "configs/training/ensemble_config_reordered.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check UNETR parameters
    unetr_params = config['models']['unetr_compact']['params']
    hidden_size = unetr_params['hidden_size']
    
    # Check if hidden_size is divisible by 6 (required for 3D position embedding)
    if hidden_size % 6 == 0:
        print(f"✅ UNETR hidden_size ({hidden_size}) is divisible by 6")
        return True
    else:
        print(f"❌ UNETR hidden_size ({hidden_size}) not divisible by 6 - will cause pos embedding error")
        return False
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("🚀 GPU Training Configuration Verification")
    print("=" * 60)
    
    checks = [
        verify_config(),
        verify_dataset_size(),
        verify_mixed_precision(),
        verify_model_parameters()
    ]
    
    # Count successful checks
    passed_checks = sum(1 for check in checks if check is not False)
    total_checks = len(checks)
    
    if all(check is not False for check in checks):
        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED!")
        print("🚀 Ready for GPU training on RTX 4080!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Transfer this code to your GPU machine")
        print("2. Ensure CUDA and PyTorch GPU support is installed")
        print("3. Run: python scripts/train_ensemble_enhanced.py --config configs/training/ensemble_config_reordered.yaml")
        return True
    else:
        print("\n" + "=" * 60)
        print(f"❌ {total_checks - passed_checks} CHECKS FAILED")
        print("Please fix the issues above before running on GPU")
        print("=" * 60)
        return False

if __name__ == "__main__":
    main()
