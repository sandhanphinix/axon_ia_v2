# Enhanced Ensemble Training System - Small Lesion Optimization

## Overview

This document describes the comprehensive ensemble training system designed specifically for improving small lesion detection in medical image segmentation. The system incorporates state-of-the-art techniques and has been optimized based on the post-training analysis recommendations.

## Key Features

### 🎯 **Small Lesion Optimization**
- **Advanced Loss Functions**: Multiple specialized loss functions optimized for small lesion detection
- **Class Balancing**: Weighted sampling to boost small lesion examples
- **Curriculum Learning**: Progressive training starting with easier samples
- **Volume Ratio Metric**: Implementation of the challenge-specific volume ratio metric

### 🏗️ **5-Model Ensemble Architecture**
1. **SwinUNETR Large**: Transformer-based with hierarchical attention
2. **UNETR Enhanced**: Pure vision transformer encoder
3. **SegResNet with Attention**: Residual network with spatial attention
4. **ResUNet with Attention Gates**: U-Net with residual connections and attention
5. **MultiScale DenseUNet**: Dense connections with pyramid pooling

### 📊 **Advanced Training Strategies**
- **5-Fold Cross-Validation**: Stratified splits based on lesion volume
- **Enhanced Data Augmentation**: 12+ augmentation techniques
- **Multi-Scale Training**: Training at different resolutions
- **Test-Time Augmentation**: Multiple predictions averaged during inference
- **Deep Supervision**: Auxiliary losses at multiple scales

### 🧠 **Specialized Loss Functions**
- **Adaptive Focal Dice**: Dynamically adjusted focus on hard examples
- **Small Lesion Loss**: Specialized weighting for small lesions
- **Combo Loss V2**: Combination of Dice, Focal, and Boundary losses
- **Tversky Focal**: Handles class imbalance with precision/recall control
- **Unified Focal Loss**: State-of-the-art unified approach

## File Structure

```
├── configs/training/ensemble_config.yaml       # Main configuration
├── scripts/
│   ├── train_ensemble_enhanced.py             # Main training script
│   └── run_ensemble_training.py               # Easy runner script
├── axon_ia/
│   ├── models/ensemble.py                     # Enhanced model architectures
│   ├── losses/advanced_losses.py             # Specialized loss functions
│   └── evaluation/metrics.py                 # Includes volume ratio metric
└── docs/ensemble_training_guide.md           # This document
```

## Configuration Highlights

### Model-Specific Optimizations

Each model in the ensemble has been individually optimized:

#### SwinUNETR Large
- **Feature Size**: 96 (increased from default 48)
- **Window Attention**: 7×7×7 windows for local feature extraction
- **Loss**: Adaptive Focal Dice with label smoothing
- **Optimizer**: AdamW with cosine warmup scheduling

#### UNETR Enhanced
- **Architecture**: Pure transformer with 12 layers
- **Attention Heads**: 12 heads for rich feature representation
- **Loss**: Small Lesion Loss with boundary component
- **Optimizer**: AdamW with polynomial decay

#### SegResNet with Attention
- **Spatial Attention**: Enhanced boundary detection
- **Deep Supervision**: Multi-scale loss computation
- **Loss**: Tversky Focal for class imbalance handling
- **Optimizer**: SGD with Nesterov momentum

#### ResUNet with Attention Gates
- **Attention Gates**: Focus on relevant features
- **Squeeze-Excitation**: Channel attention mechanism
- **Loss**: Combo Loss V2 with boundary preservation
- **Optimizer**: Adam with plateau-based scheduling

#### MultiScale DenseUNet
- **Dense Connections**: Feature reuse across scales
- **Pyramid Pooling**: Multi-scale context aggregation
- **Loss**: Unified Focal Loss
- **Optimizer**: RAdam with cyclic scheduling

### Data Augmentation Strategy

The enhanced augmentation pipeline includes:

**Spatial Transformations**:
- Rotation: ±15 degrees
- Scaling: 0.85-1.15x
- Shearing: ±5 degrees
- Translation: ±10 voxels

**Intensity Augmentations**:
- Brightness: ±15%
- Contrast: 0.85-1.15x
- Gamma correction: 0.8-1.2
- Gaussian noise: σ=0.08
- Gaussian blur: σ=0.5-1.0

**Advanced Techniques**:
- Elastic deformation
- MixUp (α=0.2)
- CutMix (α=1.0)
- Mosaic augmentation
- Copy-paste augmentation

### Small Lesion Detection Features

#### Curriculum Learning
- **Warmup Period**: 20 epochs of easier samples
- **Difficulty Metric**: Based on lesion volume
- **Easy Sample Ratio**: 70% during warmup

#### Class Balancing
- **Small Lesion Threshold**: <100 voxels
- **Boost Factor**: 2.5x sampling weight
- **Hard Negative Mining**: Focus on difficult negative examples

#### Loss Function Optimization
- **Size-Adaptive Weighting**: Higher weights for smaller lesions
- **Boundary Preservation**: Edge-aware loss components
- **Multi-Scale Supervision**: Losses at multiple resolutions

## Usage Instructions

### Quick Start (Recommended)

```bash
# Navigate to project directory
cd c:\development\axon_ia_v2

# Run the interactive training launcher
python scripts\run_ensemble_training.py
```

### Manual Training Commands

```bash
# Train full ensemble (all models, all folds)
python scripts\train_ensemble_enhanced.py --config configs\training\ensemble_config.yaml

# Train specific model across all folds
python scripts\train_ensemble_enhanced.py --config configs\training\ensemble_config.yaml --model swinunetr_large

# Train specific model and fold
python scripts\train_ensemble_enhanced.py --config configs\training\ensemble_config.yaml --model swinunetr_large --fold 0

# Validate configuration
python scripts\train_ensemble_enhanced.py --config configs\training\ensemble_config.yaml --dry-run

# Enable debug logging
python scripts\train_ensemble_enhanced.py --config configs\training\ensemble_config.yaml --debug
```

### Expected Training Time

- **Single Model, Single Fold**: ~4-6 hours (depending on GPU)
- **Single Model, All Folds**: ~20-30 hours
- **Full Ensemble**: ~100-150 hours

### Hardware Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (RTX 4090, A100, etc.)
- **RAM**: ≥32GB system memory
- **Storage**: ≥100GB free space for outputs
- **CUDA**: Compatible CUDA installation

## Key Improvements Over First Round

### 🔬 **Small Lesion Detection**
1. **Specialized Loss Functions**: Multiple loss functions specifically designed for small lesion detection
2. **Adaptive Weighting**: Dynamic sample weighting based on lesion size
3. **Boundary Preservation**: Edge-aware loss components to improve segmentation boundaries
4. **Multi-Scale Supervision**: Training at multiple resolutions to capture lesions of different sizes

### 📈 **Training Strategies**
1. **Curriculum Learning**: Progressive difficulty to improve convergence
2. **Class Balancing**: Proper handling of class imbalance typical in medical imaging
3. **Enhanced Augmentation**: 12+ augmentation techniques for better generalization
4. **Cross-Validation**: Stratified 5-fold CV for robust evaluation

### 🎯 **Model Architecture**
1. **Diverse Ensemble**: 5 different architectures with complementary strengths
2. **Attention Mechanisms**: Spatial and channel attention in multiple models
3. **Multi-Scale Features**: Pyramid pooling and dense connections
4. **Deep Supervision**: Auxiliary losses at multiple network depths

### 📊 **Evaluation Metrics**
1. **Volume Ratio**: Implementation of challenge-specific volume ratio metric
2. **Per-Lesion Analysis**: Metrics computed for different lesion sizes
3. **Cross-Validation**: Proper validation using val split, test for final evaluation

## Expected Improvements

Based on the implemented optimizations, we expect:

### Performance Gains
- **Dice Score**: +5-10% improvement on small lesions
- **Volume Ratio**: Significant improvement in volume estimation accuracy
- **False Positive Rate**: Reduced due to better boundary detection
- **Sensitivity**: Higher detection rate for small lesions

### Robustness
- **Generalization**: Better performance on unseen data due to enhanced augmentation
- **Stability**: More consistent results across different patients/scans
- **Confidence**: Better uncertainty estimation through ensemble voting

## Next Steps

After training completion:

1. **Model Evaluation**: Run comprehensive evaluation on test set
2. **Ensemble Analysis**: Compare individual models vs ensemble performance
3. **Lesion Size Analysis**: Detailed analysis of performance by lesion size
4. **Clinical Validation**: Validate results with medical experts
5. **Deployment**: Optimize models for clinical deployment

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in config
- Enable gradient accumulation
- Use mixed precision training

**Slow Training**:
- Increase num_workers for data loading
- Enable pin_memory
- Use faster storage (SSD)

**Poor Convergence**:
- Adjust learning rates per model
- Enable/disable curriculum learning
- Modify loss function weights

### Support

For issues or questions:
1. Check the logs in the output directory
2. Validate configuration with `--dry-run`
3. Enable debug mode with `--debug`
4. Review this documentation

## Conclusion

This enhanced ensemble training system represents a significant advancement over the initial training approach. By incorporating specialized techniques for small lesion detection, advanced training strategies, and a diverse set of optimized models, we expect substantial improvements in segmentation performance, particularly for the challenging task of detecting small lesions in medical images.

The system is designed to be both powerful and user-friendly, with comprehensive configuration options and easy-to-use runner scripts. The modular design allows for easy experimentation with different components while maintaining reproducibility and reliability.
