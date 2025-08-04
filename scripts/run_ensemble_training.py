#!/usr/bin/env python
"""
Enhanced runner script for ensemble training with small lesion optimization.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run ensemble training with the enhanced script."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs" / "training" / "ensemble_config.yaml"
    train_script = script_dir / "train_ensemble_enhanced.py"
    
    # Check if files exist
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return 1
    
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        return 1
    
    # Setup environment
    python_executable = sys.executable
    
    print("=" * 80)
    print("ENSEMBLE TRAINING RUNNER - SMALL LESION OPTIMIZATION")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Script: {train_script}")
    print(f"Python: {python_executable}")
    print()
    print("🖥️  CPU-OPTIMIZED CONFIGURATION:")
    print("• Optimized for CPU-only training (no GPU required)")
    print("• Reduced model complexity for faster training")
    print("• Limited memory usage and batch sizes")
    print("• Progress bars and status indicators included")
    print()
    print("Features:")
    print("• 5 state-of-the-art models optimized for small lesions")
    print("• Advanced loss functions (Focal, Tversky, Boundary)")
    print("• Curriculum learning and class balancing")
    print("• 5-fold cross-validation")
    print("• Enhanced data augmentation")
    print("• Volume ratio metric calculation")
    print("• Real-time progress tracking with progress bars")
    print()
    
    # Ask user for training options
    print("Training options:")
    print("1. Train all models, all folds (full ensemble) - RECOMMENDED")
    print("2. Train specific model, all folds")
    print("3. Train specific model, specific fold")
    print("4. Dry run (validate config only)")
    
    choice = input("Select option (1-4): ").strip()
    
    # Build command
    cmd = [python_executable, str(train_script), "--config", str(config_path)]
    
    if choice == "1":
        print("Training full ensemble with all 5 models...")
        print("This will train:")
        print("• SwinUNETR Large (Transformer-based)")
        print("• UNETR Enhanced (Pure transformer)")
        print("• SegResNet with Attention")
        print("• ResUNet with Attention Gates")
        print("• MultiScale DenseUNet")
    elif choice == "2":
        models = ["swinunetr_large", "unetr_enhanced", "segresnet_attention", 
                 "resunet_attention", "multiscale_denseunet"]
        print("Available models:")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}")
        
        model_choice = input("Select model (1-5): ").strip()
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(models):
                cmd.extend(["--model", models[model_idx]])
                print(f"Training {models[model_idx]} across all 5 folds...")
            else:
                print("Invalid model choice")
                return 1
        except ValueError:
            print("Invalid input")
            return 1
    elif choice == "3":
        models = ["swinunetr_large", "unetr_enhanced", "segresnet_attention", 
                 "resunet_attention", "multiscale_denseunet"]
        print("Available models:")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}")
        
        model_choice = input("Select model (1-5): ").strip()
        fold_choice = input("Select fold (0-4): ").strip()
        
        try:
            model_idx = int(model_choice) - 1
            fold_idx = int(fold_choice)
            
            if 0 <= model_idx < len(models) and 0 <= fold_idx <= 4:
                cmd.extend(["--model", models[model_idx], "--fold", str(fold_idx)])
                print(f"Training {models[model_idx]} for fold {fold_idx}...")
            else:
                print("Invalid choice")
                return 1
        except ValueError:
            print("Invalid input")
            return 1
    elif choice == "4":
        cmd.append("--dry-run")
        print("Running configuration validation...")
    else:
        print("Invalid choice")
        return 1
    
    # Enable debug mode option
    debug = input("Enable debug mode? (y/N): ").strip().lower()
    if debug == 'y':
        cmd.append("--debug")
    
    print()
    print("Command to execute:")
    print(" ".join(cmd))
    print()
    
    if choice != "4":
        confirm = input("Start training? (Y/n): ").strip().lower()
        if confirm == 'n':
            print("Training cancelled.")
            return 0
    
    # Run the training
    try:
        print("Starting training...")
        print("-" * 80)
        subprocess.run(cmd, check=True)
        print("-" * 80)
        print("=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Next steps:")
        print("1. Check the output directory for trained models")
        print("2. Run post-training analysis to evaluate performance")
        print("3. Use the ensemble for inference on test data")
        return 0
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print("=" * 80)
        print(f"TRAINING FAILED WITH ERROR CODE: {e.returncode}")
        print("=" * 80)
        print("Check the logs above for error details.")
        return e.returncode
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
