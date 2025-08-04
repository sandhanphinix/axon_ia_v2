# Ensemble Training Fixes Summary

## Issues Fixed

### 1. UNETR Model Shape Mismatch Error
**Problem**: UNETR was getting tensor shape `[1, 64, 768]` when expecting 4D/5D tensor for `conv_transpose3d`

**Fix**:
- Updated the UNETR forward method to properly reshape hidden states from `(B, num_patches, hidden_size)` to `(B, hidden_size, D, H, W)`
- Added robust layer indexing to handle different ViT configurations
- Ensured ViT has enough layers for skip connections
- Fixed patch dimension calculations

**Files Changed**: `axon_ia/models/unetr.py`

### 2. SwinUNETR Performance Drop (0.56 → 0.3 DICE)
**Problem**: Significant performance degradation compared to standalone model

**Fixes Applied**:
- **Image Size**: Increased from 64³ to 96³ (still CPU-friendly but better resolution)
- **Learning Rate**: Increased from 1e-5 to 2e-4 for faster convergence
- **Boundary Loss**: Added small boundary loss (0.1) for better edge detection
- **Training Duration**: Increased max epochs from 50 to 75
- **Batch Accumulation**: Increased from 4 to 6 for more stable gradients
- **Preprocessing**: Updated ROI size to match image size (96³)

**Expected Result**: Should achieve closer to original 0.56 DICE performance

### 3. Checkpoint Resume Functionality
**Problem**: No way to resume training after failures, wasting days of computation

**Fix**:
- Added automatic checkpoint detection and resuming
- Checkpoints saved every 5 epochs with full training state
- Resume from latest checkpoint automatically
- Preserves epoch count, best DICE, patience counter, optimizer state

**Files Changed**: `scripts/train_ensemble_enhanced.py`

### 4. Model Robustness Improvements
**Problem**: Risk of model failures after days of training

**Fixes for All Models**:

#### SegResNet:
- Changed normalization from batch to instance for stability
- Reduced dropout from 0.1 to 0.05
- Simplified block depths: [1,2,2,2] instead of [1,2,2,4]
- Increased learning rate to 1e-4
- Disabled deep supervision for stability

#### ResUNet:
- Reduced feature complexity: [24,48,96,192] vs [32,64,128,256]  
- Very low dropout (0.05) to prevent convergence issues
- Disabled deep supervision for stability
- Increased learning rate to 1e-4

#### MultiScale UNet:
- Disabled multiscale features and pyramid pooling (complexity reduction)
- Simplified num_layers: [2,2,3,3] vs [2,3,4,5]
- Conservative base_features (20) and growth_rate (8)
- Disabled deep supervision
- Increased learning rate to 1e-4

#### UNETR:
- Increased feature_size from 16 to 24
- Updated to use 96³ image size
- Kept deep supervision disabled for compatibility

#### SwinUNETR:
- Improved configuration as detailed above

## Configuration Changes

### Training Parameters:
- **Max Epochs**: 50 → 75 (more training time)
- **Early Stopping Patience**: 15 → 20 
- **Gradient Accumulation**: 4 → 6 batches
- **Image Size**: 64³ → 96³ (all models)
- **Curriculum Learning**: More conservative (15 warmup epochs, 0.6 easy ratio)
- **Class Balancing**: More aggressive (3.0 boost for small lesions)

### Loss Functions:
- **SwinUNETR**: Added boundary loss (0.1 weight)
- **Other Models**: Reduced focal weights to 0.3 for stability
- **All**: Consistent combo_loss_v2 usage

### Learning Rates:
- **SwinUNETR**: 1e-5 → 2e-4 (needs aggressive learning)
- **UNETR**: Kept at 1e-5 (large model, needs conservative LR)
- **Others**: 1e-5 → 1e-4 (balanced for convergence)

## Expected Results

### Performance:
- **SwinUNETR**: Should achieve 0.45-0.56 DICE (closer to original performance)
- **Other Models**: Should achieve 0.25-0.40 DICE (stable convergence)
- **Ensemble**: Should achieve 0.50+ DICE through model diversity

### Robustness:
- **Checkpoint Resume**: Automatic recovery from failures
- **Model Stability**: Reduced risk of NaN losses or convergence failures
- **Memory Usage**: Still CPU-compatible with conservative batch sizes

### Training Time:
- **Per Model**: ~3-4 days per fold (75 epochs instead of 50)
- **Total Ensemble**: ~60-75 days for all 5 models × 5 folds
- **Interruption Recovery**: Can resume from any checkpoint

## Usage

### Normal Training (New Reordered Config):
```bash
python scripts/train_ensemble_enhanced.py --config configs/training/ensemble_config_reordered.yaml
```

### Resume from Specific Model/Fold:
```bash
python scripts/train_ensemble_enhanced.py --config configs/training/ensemble_config_reordered.yaml --model unetr_compact --fold 2
```

### Test Configuration:
```bash
python scripts/train_ensemble_enhanced.py --config configs/training/ensemble_config_reordered.yaml --dry-run
```

## Files Modified

1. **`configs/training/ensemble_config_reordered.yaml`** - NEW: Reordered models config with fixes
2. **`configs/training/ensemble_config.yaml`** - Original config (still available)
3. **`scripts/train_ensemble_enhanced.py`** - Added checkpoint resume functionality  
4. **`axon_ia/models/unetr.py`** - Fixed tensor shape issues

## Monitoring

### Checkpoints:
- Location: `{output_dir}/checkpoints/{model_name}_fold_{fold}/`
- Frequency: Every 5 epochs
- Content: Full training state (model, optimizer, metrics, config)

### Logs:
- TensorBoard: `logs/ensemble_v2/`
- Console: Real-time training progress with DICE scores

### Key Metrics to Watch:
- **DICE Score**: Should steadily increase, target >0.4 for SwinUNETR
- **Loss Convergence**: Should decrease without NaN values
- **Memory Usage**: Should stay within CPU limits
- **Training Speed**: ~8-10 seconds per batch

## Troubleshooting

### If Training Fails:
1. Check latest checkpoint in output directory
2. Resume will happen automatically on next run
3. Monitor memory usage - reduce batch accumulation if needed
4. Check logs for specific error messages

### If Performance is Poor:
1. Verify data loading is working correctly
2. Check loss function weights
3. Consider adjusting learning rates
4. Increase training epochs if convergence is slow

### If Memory Issues:
1. Reduce image size back to 64³
2. Reduce gradient accumulation steps
3. Disable deep supervision on more models
4. Reduce model feature sizes further

## Expected Timeline

**NEW TRAINING ORDER (Testing fixes first, SwinUNETR last):**
- **Day 1-4**: UNETR Fold 1 (test the fixed shape mismatch)
- **Day 5-8**: UNETR Fold 2
- **Day 9-12**: UNETR Fold 3
- **Day 13-16**: UNETR Fold 4
- **Day 17-20**: UNETR Fold 5
- **Day 21-24**: SegResNet Fold 1 (test robustness improvements)
- **Day 25-28**: SegResNet Fold 2
- **Day 29-32**: SegResNet Fold 3
- **Day 33-36**: SegResNet Fold 4
- **Day 37-40**: SegResNet Fold 5
- **Day 41-44**: ResUNet Fold 1 (test stability fixes)
- **Day 45-48**: ResUNet Fold 2
- **Day 49-52**: ResUNet Fold 3
- **Day 53-56**: ResUNet Fold 4
- **Day 57-60**: ResUNet Fold 5  
- **Day 61-64**: MultiScale UNet Fold 1 (test simplified architecture)
- **Day 65-68**: MultiScale UNet Fold 2
- **Day 69-72**: MultiScale UNet Fold 3
- **Day 73-76**: MultiScale UNet Fold 4
- **Day 77-80**: MultiScale UNet Fold 5
- **Day 81-84**: SwinUNETR Fold 1 (known working, improved config)
- **Day 85-88**: SwinUNETR Fold 2
- **Day 89-92**: SwinUNETR Fold 3
- **Day 93-96**: SwinUNETR Fold 4
- **Day 97-100**: SwinUNETR Fold 5

**Strategy**: Test all the uncertain models first to validate the fixes. If any fail, you'll know early and can adjust. SwinUNETR is saved for last as the reliable backup.
