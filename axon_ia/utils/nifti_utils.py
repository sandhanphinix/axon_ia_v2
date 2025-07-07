"""
Utilities for working with NIfTI medical image files.

This module provides functions for loading, saving, and processing
NIfTI files commonly used in medical imaging.
"""

import os
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict

import nibabel as nib
import numpy as np
import torch


def load_nifti(
    filepath: Union[str, Path],
    return_meta: bool = False,
    dtype: Optional[type] = None,
    normalize: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Load a NIfTI file.
    
    Args:
        filepath: Path to the NIfTI file
        return_meta: Whether to return metadata
        dtype: Data type to cast to
        normalize: Whether to normalize intensity values
        
    Returns:
        Image array and optional metadata
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    
    if dtype is not None:
        data = data.astype(dtype)
    
    if normalize:
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    if return_meta:
        meta = {
            'affine': img.affine,
            'header': img.header,
            'shape': img.shape,
            'zooms': img.header.get_zooms()
        }
        return data, meta
    else:
        return data


def save_nifti(
    data: Union[np.ndarray, torch.Tensor],
    filepath: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None,
):
    """
    Save data as a NIfTI file.
    
    Args:
        data: Image data
        filepath: Output path
        affine: Affine transformation matrix
        header: NIfTI header
    """
    # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Use identity matrix if no affine provided
    if affine is None:
        affine = np.eye(4)
    
    # Create and save NIfTI image
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, filepath)


def resample_nifti(
    img: nib.Nifti1Image,
    target_shape: Optional[Tuple[int, ...]] = None,
    target_spacing: Optional[Tuple[float, ...]] = None,
    interpolation: str = 'linear'
) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to a new shape or spacing.
    
    Args:
        img: Input NIfTI image
        target_shape: Target shape
        target_spacing: Target spacing in mm
        interpolation: Interpolation method ('linear', 'nearest')
        
    Returns:
        Resampled NIfTI image
    """
    import SimpleITK as sitk
    
    # Convert to SimpleITK image
    data = img.get_fdata()
    original_spacing = img.header.get_zooms()
    original_affine = img.affine
    
    sitk_img = sitk.GetImageFromArray(data)
    sitk_img.SetSpacing(original_spacing)
    
    # Calculate target spacing from shape if not provided
    if target_spacing is None and target_shape is not None:
        current_shape = img.shape
        target_spacing = tuple(
            orig_spacing * orig_shape / target
            for orig_spacing, orig_shape, target in zip(original_spacing, current_shape, target_shape)
        )
    
    # Determine interpolation method
    if interpolation == 'linear':
        interp_method = sitk.sitkLinear
    elif interpolation == 'nearest':
        interp_method = sitk.sitkNearestNeighbor
    else:
        interp_method = sitk.sitkLinear
    
    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp_method)
    resampler.SetOutputSpacing(target_spacing)
    
    # Calculate size based on target spacing
    original_size = sitk_img.GetSize()
    original_spacing = sitk_img.GetSpacing()
    output_size = [
        int(round(orig_size * orig_spacing / target))
        for orig_size, orig_spacing, target in zip(original_size, original_spacing, target_spacing)
    ]
    resampler.SetSize(output_size)
    
    # Set default parameters
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    
    # Resample
    resampled_sitk = resampler.Execute(sitk_img)
    
    # Convert back to numpy and create new NIfTI
    resampled_data = sitk.GetArrayFromImage(resampled_sitk)
    
    # Adjust affine for new spacing
    scale_factor = np.ones(4)
    scale_factor[:3] = np.array(target_spacing) / np.array(original_spacing[:3])
    scaling_matrix = np.diag(scale_factor)
    new_affine = original_affine @ scaling_matrix
    
    # Create new NIfTI image
    new_img = nib.Nifti1Image(resampled_data, new_affine, img.header)
    new_img.header.set_zooms(target_spacing)
    
    return new_img


def get_brain_mask(
    img_data: np.ndarray,
    threshold: float = 0.01,
    min_size: int = 1000,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Create a binary brain mask from an MRI image.
    
    Args:
        img_data: Image data
        threshold: Threshold for binarization
        min_size: Minimum connected component size
        fill_holes: Whether to fill holes in the mask
        
    Returns:
        Binary brain mask
    """
    from scipy import ndimage
    
    # Normalize data
    normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    
    # Create initial mask
    mask = normalized > threshold
    
    # Remove small connected components
    labeled, num_features = ndimage.label(mask)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    
    # Keep only large components
    small_components = component_sizes < min_size
    remove_indices = np.where(small_components)[0] + 1
    for idx in remove_indices:
        mask[labeled == idx] = 0
    
    # Fill holes if requested
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    
    return mask


def combine_masks(masks: List[np.ndarray], method: str = 'union') -> np.ndarray:
    """
    Combine multiple binary masks.
    
    Args:
        masks: List of binary masks
        method: Combination method ('union', 'intersection', 'majority')
        
    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("No masks provided")
    
    if method == 'union':
        return np.any(masks, axis=0).astype(np.int8)
    
    elif method == 'intersection':
        return np.all(masks, axis=0).astype(np.int8)
    
    elif method == 'majority':
        return (np.sum(masks, axis=0) > len(masks) // 2).astype(np.int8)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_volume_stats(
    mask: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None
) -> Dict[str, float]:
    """
    Compute volume statistics for a binary mask.
    
    Args:
        mask: Binary mask
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary of statistics
    """
    # Ensure binary mask
    binary_mask = mask > 0
    
    # Get voxel count
    voxel_count = np.sum(binary_mask)
    
    # Default stats
    stats = {
        'voxel_count': voxel_count,
        'volume_voxels': voxel_count,
    }
    
    # Compute physical volume if spacing is provided
    if spacing is not None:
        voxel_volume = np.prod(spacing)
        physical_volume = voxel_count * voxel_volume
        stats['voxel_volume_mm3'] = voxel_volume
        stats['volume_mm3'] = physical_volume
        stats['volume_ml'] = physical_volume / 1000.0  # Convert to milliliters
    
    return stats


def extract_roi(
    img_data: np.ndarray,
    mask: np.ndarray,
    margin: int = 5
) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
    """
    Extract a region of interest based on a mask.
    
    Args:
        img_data: Image data
        mask: Binary mask
        margin: Margin to add around the ROI
        
    Returns:
        Tuple of (cropped image, crop indices)
    """
    # Find bounding box of the mask
    indices = np.where(mask > 0)
    if not indices[0].size:
        return img_data, ((0, img_data.shape[0]), (0, img_data.shape[1]), (0, img_data.shape[2]))
    
    # Get min/max indices with margin
    min_indices = [max(0, int(np.min(idx)) - margin) for idx in indices]
    max_indices = [min(img_data.shape[i], int(np.max(indices[i])) + margin + 1) for i in range(len(indices))]
    
    # Create crop indices
    crop_indices = tuple((min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices))
    
    # Crop image
    cropped_img = img_data[
        crop_indices[0][0]:crop_indices[0][1],
        crop_indices[1][0]:crop_indices[1][1],
        crop_indices[2][0]:crop_indices[2][1]
    ]
    
    return cropped_img, crop_indices


def apply_window_level(
    img_data: np.ndarray,
    window: float,
    level: float
) -> np.ndarray:
    """
    Apply window/level transformation to image data.
    
    Args:
        img_data: Image data
        window: Window width
        level: Window center
        
    Returns:
        Windowed image
    """
    # Calculate min and max values
    min_value = level - window / 2
    max_value = level + window / 2
    
    # Apply windowing
    windowed = np.clip(img_data, min_value, max_value)
    
    # Normalize to [0, 1]
    normalized = (windowed - min_value) / window
    
    return normalized