# Data Setup Guide for Ensemble Training

## Expected Directory Structure

The ensemble training system expects your data to be organized in the following structure:

```
dataset_path/
├── train/                    # 334 patient folders
│   ├── 001/
│   │   ├── b0_001.nii.gz           # Training modality
│   │   ├── b1000_001.nii.gz        # Training modality  
│   │   ├── flair_001.nii.gz        # Training modality
│   │   ├── T2Star_001.nii.gz       # Training modality
│   │   ├── perfroi_001.nii.gz      # Target: brain lesions
│   │   └── eloquentareas_001.nii.gz # Target: eloquent areas
│   ├── 002/
│   │   ├── b0_002.nii.gz
│   │   ├── b1000_002.nii.gz
│   │   ├── flair_002.nii.gz
│   │   ├── T2Star_002.nii.gz
│   │   ├── perfroi_002.nii.gz
│   │   └── eloquentareas_002.nii.gz
│   └── ...
└── test/                     # 109 patient folders
    ├── 335/
    │   ├── b0_335.nii.gz
    │   ├── b1000_335.nii.gz
    │   ├── flair_335.nii.gz
    │   ├── T2Star_335.nii.gz
    │   ├── perfroi_335.nii.gz
    │   └── eloquentareas_335.nii.gz
    └── ...
```

## File Naming Convention

**Important**: Files must follow the exact naming pattern: `<modality>_<folder_name>.nii.gz`

Where:
- `<modality>` is one of: b0, b1000, flair, T2Star, perfroi, eloquentareas
- `<folder_name>` is the numeric folder name (e.g., 001, 002, 335, etc.)

### Training Modalities:
- `b0_<folder_name>.nii.gz` - B0 diffusion weighted image
- `b1000_<folder_name>.nii.gz` - B1000 diffusion weighted image  
- `flair_<folder_name>.nii.gz` - FLAIR (Fluid Attenuated Inversion Recovery)
- `T2Star_<folder_name>.nii.gz` - T2* weighted image

### Target Segmentations:
- `perfroi_<folder_name>.nii.gz` - Brain lesion segmentation masks
- `eloquentareas_<folder_name>.nii.gz` - Eloquent area segmentation masks

## Configuration

Update the configuration file to specify which target you want to train for:

```yaml
data:
  dataset_path: "C:/development/data/axon_ia"
  modalities: ["b0", "b1000", "flair", "T2Star"]  # Training images
  target: "perfroi"  # For brain lesion segmentation
  # target: "eloquentareas"  # For eloquent area segmentation
```

## Troubleshooting Common Issues

### 1. "Training dataset is empty" Error

**Cause**: The system cannot find any matching image-mask pairs in the training directory.

**Solutions**:
- Check that `dataset_path/train/images/` and `dataset_path/train/masks/` exist
- Verify that image and mask files have identical names
- Ensure files are in supported formats (`.nii.gz` or `.nii`)
- Check file permissions (files should be readable)

### 2. "Validation dataset is empty" Warning

**Cause**: No validation data found.

**Solutions**:
- Create `dataset_path/validation/` or `dataset_path/val/` directory
- Add images and masks subdirectories with matching files
- If you don't have separate validation data, the system will automatically create an 80/20 split from training data

### 3. "Cannot create cross-validation splits" Error

**Cause**: Not enough samples for the requested number of folds.

**Solutions**:
- Reduce the number of folds in your config: `cross_validation.n_folds: 3`
- Add more training samples
- Disable cross-validation: `cross_validation.enabled: false`

### 4. File Format Issues

**Supported formats**:
- NIfTI: `.nii`, `.nii.gz` (recommended for compression)
- DICOM: Not directly supported (convert to NIfTI first)

**Conversion tools**:
- [dcm2niix](https://github.com/rordenlab/dcm2niix) for DICOM to NIfTI
- [3D Slicer](https://www.slicer.org/) for format conversion
- [ITK-SNAP](http://www.itksnap.org/) for viewing and conversion

## Minimal Working Example

For testing, create this minimal structure:

```bash
# Windows PowerShell
mkdir C:\development\data\axon_ia\train\images
mkdir C:\development\data\axon_ia\train\masks
mkdir C:\development\data\axon_ia\validation\images
mkdir C:\development\data\axon_ia\validation\masks

# Copy your .nii.gz files to the appropriate directories
# Ensure matching names between images and masks
```

## Data Quality Recommendations

### For Medical Image Segmentation:

1. **Image Quality**:
   - Consistent voxel spacing across all images
   - Proper intensity normalization
   - Remove any corrupted or incomplete scans

2. **Mask Quality**:
   - Binary masks (0 for background, 1 for lesion)
   - Accurate annotations by medical professionals
   - Consistent annotation guidelines across all cases

3. **Dataset Size**:
   - **Minimum**: 50-100 cases for basic training
   - **Recommended**: 200+ cases for robust models
   - **Ideal**: 500+ cases for production-ready models

4. **Class Balance**:
   - For small lesion detection, ensure adequate representation of small lesions
   - The system includes automatic class balancing for small lesions
   - Monitor lesion size distribution in your dataset

## Advanced Configuration

### For Small Lesion Detection:

```yaml
global:
  training:
    class_balancing:
      enabled: true
      small_lesion_threshold: 100  # voxels
      small_lesion_boost: 2.5      # sampling boost factor
```

### For Large Datasets:

```yaml
data:
  cache_rate: 0.5        # Increase if you have more RAM
  num_workers: 4         # Increase for faster loading
  pin_memory: true       # Enable if using GPU
```

### For Memory-Constrained Systems:

```yaml
data:
  cache_rate: 0.1        # Reduce caching
  num_workers: 1         # Single worker
  pin_memory: false      # Disable pinning

global:
  training:
    batch_size: 1        # Smallest possible batch
```

## Getting Help

If you continue to have issues:

1. **Check the console output** for detailed error messages
2. **Verify file permissions** (especially on Windows)
3. **Test with a small subset** of your data first
4. **Check available disk space** and memory
5. **Review the training logs** for specific error details

The ensemble training system includes comprehensive error checking and will provide specific guidance when issues are detected.
