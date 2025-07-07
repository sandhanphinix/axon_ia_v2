# Data Format Documentation

## Overview

Axon IA expects data in a specific format to ensure consistent training and evaluation. This document outlines the required directory structure, file formats, and conventions used throughout the framework.

## Directory Structure

Axon IA uses a hierarchical directory structure organized by dataset splits:

```
<root_data_dir>/
    train/
        case_001/
            flair.nii.gz
            t1.nii.gz
            t2.nii.gz
            dwi.nii.gz
            mask.nii.gz
        case_002/
            ...
    val/
        case_101/
            flair.nii.gz
            t1.nii.gz
            t2.nii.gz
            dwi.nii.gz
            mask.nii.gz
        ...
    test/
        case_201/
            flair.nii.gz
            t1.nii.gz
            t2.nii.gz
            dwi.nii.gz
            mask.nii.gz  # Optional for test
        ...
```

- Each `case_xxx` folder contains NIfTI files for each modality and the segmentation mask.
- Modalities are configurable (default: flair, t1, t2, dwi).
- The mask file should be a binary or multi-class segmentation mask.

## File Naming Conventions
- Modality files: `<modality>.nii.gz` (e.g., `flair.nii.gz`)
- Target mask: `mask.nii.gz`
- All files must be in NIfTI format (`.nii` or `.nii.gz`).

## Metadata
- Preprocessing generates a `preprocessing_metadata.json` file in the output directory, containing information about spacing, normalization, and case splits.
- Optionally, per-case metadata can be included (e.g., `train_metadata.json`).

## Preprocessing Steps
- **Resampling**: All images are resampled to a target spacing (default: 1.0x1.0x1.0 mm)
- **Orientation**: Standardized to a common orientation (default: RAS)
- **Normalization**: Intensity normalization (z-score, percentile, or min-max)
- **Cropping**: Foreground cropping to remove background

## Augmentation
- Augmentation is applied during training (random rotations, scaling, flipping, noise, gamma correction, elastic deformations, etc.)
- See configuration files for details.

## Output
- Processed data is saved in the specified output directory, preserving the split/case structure.
- Additional outputs: metadata files, logs, and visualizations.

---

For more details, see the [README](../README.md) and configuration templates in `configs/`.
