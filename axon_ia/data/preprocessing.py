"""
Preprocessing utilities for medical images.

This module provides functions for preprocessing medical images,
including resampling, normalization, and cropping.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom, binary_dilation, binary_erosion

from axon_ia.utils.logger import get_logger

logger = get_logger()


def resample_to_spacing(
    image: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    interpolation: str = 'linear',
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resample 3D image to target spacing.
    
    Args:
        image: 3D image array
        original_spacing: Original voxel spacing in mm
        target_spacing: Target voxel spacing in mm
        interpolation: Interpolation method ('linear', 'nearest')
        is_mask: Whether the image is a mask
        
    Returns:
        Resampled image
    """
    # Calculate scale factors
    scale_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    
    # Choose interpolation order
    order = 0 if is_mask else 3  # nearest or cubic
    if interpolation == 'nearest':
        order = 0
    
    # Apply zoom
    resampled = zoom(image, scale_factors, order=order)
    
    return resampled


def normalize_intensity(
    image: np.ndarray,
    mode: str = 'z_score',
    mask: Optional[np.ndarray] = None,
    percentiles: Tuple[float, float] = (1, 99),
) -> np.ndarray:
    """
    Normalize intensity values in medical images.
    
    Args:
        image: Image array to normalize
        mode: Normalization mode ('z_score', 'percentile', 'min_max')
        mask: Optional mask to focus normalization on specific region
        percentiles: Percentiles for clipping in percentile mode
        
    Returns:
        Normalized image
    """
    if mask is not None:
        # Focus on non-zero regions in mask
        values = image[mask > 0]
        if values.size == 0:
            # If mask is empty, use all values
            values = image
    else:
        values = image
    
    # Apply normalization based on mode
    if mode == 'z_score':
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return image - mean
        else:
            return (image - mean) / std
    
    elif mode == 'percentile':
        p_low, p_high = np.percentile(values, percentiles)
        
        # Clip to percentile range
        clipped = np.clip(image, p_low, p_high)
        
        # Scale to [0, 1]
        if p_high > p_low:
            return (clipped - p_low) / (p_high - p_low)
        else:
            return clipped - p_low
    
    elif mode == 'min_max':
        min_val = values.min()
        max_val = values.max()
        
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return image - min_val
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def crop_foreground(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.01,
    margin: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]:
    """
    Crop image to focus on foreground/non-zero regions.
    
    Args:
        image: Image array to crop
        mask: Optional mask to define foreground
        threshold: Threshold for foreground in image if mask not provided
        margin: Margin around foreground in voxels
        
    Returns:
        Tuple of (cropped_image, cropped_mask, crop_slices)
    """
    # Create mask if not provided
    if mask is None:
        if image.ndim == 4:  # Multi-channel
            # Use maximum across channels
            foreground = np.max(image, axis=0) > threshold
        else:
            foreground = image > threshold
    else:
        foreground = mask > 0
    
    # Find bounding box
    if not np.any(foreground):
        return image, mask, tuple(slice(None) for _ in range(image.ndim))
    
    # Get indices of non-zero elements
    indices = np.where(foreground)
    
    # Get min/max indices with margin
    mins = [max(0, int(np.min(idx)) - margin) for idx in indices]
    maxs = [min(s, int(np.max(idx)) + margin + 1) for s, idx in zip(foreground.shape, indices)]
    
    # Create crop slices
    crop_slices = tuple(slice(min_val, max_val) for min_val, max_val in zip(mins, maxs))
    
    # Apply cropping
    if image.ndim == 4:  # Multi-channel
        # Include channel dimension in crop
        crop_slices_image = (slice(None),) + crop_slices
        cropped_image = image[crop_slices_image]
    else:
        cropped_image = image[crop_slices]
    
    # Crop mask if provided
    cropped_mask = None
    if mask is not None:
        cropped_mask = mask[crop_slices]
    
    return cropped_image, cropped_mask, crop_slices


def standardize_orientation(
    image: Union[np.ndarray, sitk.Image],
    reference_orientation: str = "RAI",
) -> Union[np.ndarray, sitk.Image]:
    """
    Standardize the orientation of a medical image.
    
    Args:
        image: Image to reorient (numpy array or SimpleITK image)
        reference_orientation: Target orientation code
        
    Returns:
        Reoriented image in the same format as input
    """
    # If numpy array, convert to SimpleITK
    if isinstance(image, np.ndarray):
        as_numpy = True
        if image.ndim == 4:  # Multi-channel
            sitk_images = []
            for i in range(image.shape[0]):
                sitk_img = sitk.GetImageFromArray(image[i])
                sitk_images.append(sitk_img)
            
            # Process and convert back at the end
            result = []
            for sitk_img in sitk_images:
                reoriented = standardize_orientation(sitk_img, reference_orientation)
                result.append(sitk.GetArrayFromImage(reoriented))
            
            return np.stack(result, axis=0)
        else:
            sitk_img = sitk.GetImageFromArray(image)
    else:
        as_numpy = False
        sitk_img = image
    
    # Get current orientation
    current_direction = sitk_img.GetDirection()
    
    # Create reference direction matrix for desired orientation
    ref_image = sitk.Image([1, 1, 1], sitk.sitkUInt8)
    ref_image = sitk.DICOMOrient(ref_image, reference_orientation)
    reference_direction = ref_image.GetDirection()
    
    # Check if reorientation is needed
    if current_direction != reference_direction:
        # Reorient image
        reoriented = sitk.DICOMOrient(sitk_img, reference_orientation)
    else:
        # Already in correct orientation
        reoriented = sitk_img
    
    # Convert back to numpy if input was numpy
    if as_numpy:
        return sitk.GetArrayFromImage(reoriented)
    else:
        return reoriented


def brain_extraction(
    image: np.ndarray,
    method: str = 'threshold',
    threshold: float = 0.01,
    closing_iterations: int = 5,
) -> np.ndarray:
    """
    Extract brain region from MRI image.
    
    Args:
        image: Input brain MRI
        method: Extraction method ('threshold', 'otsu')
        threshold: Threshold value for 'threshold' method
        closing_iterations: Number of closing iterations
        
    Returns:
        Brain mask
    """
    # Handle multi-channel images
    if image.ndim == 4:
        # Use FLAIR (usually most informative for brain extraction)
        if image.shape[0] >= 4:
            img = image[0]  # Assuming FLAIR is first channel
        else:
            img = image[0]  # Use first available channel
    else:
        img = image
    
    # Method: simple thresholding
    if method == 'threshold':
        # Normalize image to [0, 1]
        normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Apply threshold
        mask = normalized > threshold
    
    # Method: Otsu thresholding
    elif method == 'otsu':
        try:
            # Convert to SimpleITK
            sitk_img = sitk.GetImageFromArray(img)
            
            # Apply Otsu thresholding
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetInsideValue(0)
            otsu_filter.SetOutsideValue(1)
            mask_sitk = otsu_filter.Execute(sitk_img)
            
            # Convert back to numpy
            mask = sitk.GetArrayFromImage(mask_sitk) > 0
        except:
            # Fallback to simple thresholding
            logger.warning("Otsu thresholding failed, falling back to simple thresholding")
            normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
            mask = normalized > threshold
    
    else:
        raise ValueError(f"Unknown brain extraction method: {method}")
    
    # Apply morphological operations to clean up the mask
    if closing_iterations > 0:
        # Closing: dilation followed by erosion
        struct = np.ones((3, 3, 3))
        mask = binary_dilation(mask, struct, iterations=closing_iterations)
        mask = binary_erosion(mask, struct, iterations=closing_iterations)
    
    # Keep only the largest connected component
    from scipy.ndimage import label
    labeled, num_components = label(mask)
    if num_components > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]
        largest_component = np.argmax(component_sizes) + 1
        mask = labeled == largest_component
    
    return mask.astype(np.float32)