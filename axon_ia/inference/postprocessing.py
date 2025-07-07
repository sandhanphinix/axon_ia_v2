"""
Postprocessing functions for medical image segmentation.

This module provides functions for postprocessing segmentation
predictions to improve quality and consistency.
"""

from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
import scipy.ndimage as ndimage

from axon_ia.utils.logger import get_logger

logger = get_logger()


def apply_threshold(
    pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply threshold to convert probability maps to binary masks.
    
    Args:
        pred: Prediction array
        threshold: Threshold value
        
    Returns:
        Binary mask
    """
    if isinstance(pred, torch.Tensor):
        return (pred > threshold).float()
    else:
        return (pred > threshold).astype(np.float32)


def remove_small_objects(
    mask: Union[np.ndarray, torch.Tensor],
    min_size: int = 100
) -> Union[np.ndarray, torch.Tensor]:
    """
    Remove small connected components from segmentation.
    
    Args:
        mask: Binary mask
        min_size: Minimum component size in voxels
        
    Returns:
        Cleaned binary mask
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        device = mask.device
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    
    # Handle multi-channel input
    if mask_np.ndim >= 4:
        # Process each channel separately
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                mask_np[b, c] = remove_small_objects(mask_np[b, c], min_size)
        
        # Convert back to tensor if needed
        if is_tensor:
            return torch.from_numpy(mask_np).to(device)
        else:
            return mask_np
    
    # Apply connected component analysis
    # Convert to binary
    binary_mask = mask_np > 0.5
    
    # Find connected components
    labeled, num_components = ndimage.label(binary_mask)
    
    if num_components == 0:
        # No components found
        return mask
    
    # Measure component sizes
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_components + 1))
    
    # Create mask of small components
    too_small = component_sizes < min_size
    
    # Remove small components
    remove_pixels = np.zeros(binary_mask.shape, bool)
    for i, small in enumerate(too_small):
        if small:
            remove_pixels[labeled == i + 1] = True
    
    # Apply mask
    result = binary_mask.astype(np.float32)
    result[remove_pixels] = 0
    
    # Convert back to tensor if needed
    if is_tensor:
        return torch.from_numpy(result).to(device)
    else:
        return result


def fill_holes(
    mask: Union[np.ndarray, torch.Tensor],
    max_hole_size: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Fill holes in segmentation mask.
    
    Args:
        mask: Binary mask
        max_hole_size: Maximum hole size to fill (None for all)
        
    Returns:
        Mask with filled holes
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        device = mask.device
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    
    # Handle multi-channel input
    if mask_np.ndim >= 4:
        # Process each channel separately
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                mask_np[b, c] = fill_holes(mask_np[b, c], max_hole_size)
        
        # Convert back to tensor if needed
        if is_tensor:
            return torch.from_numpy(mask_np).to(device)
        else:
            return mask_np
    
    # Convert to binary
    binary_mask = mask_np > 0.5
    
    if max_hole_size is None:
        # Fill all holes
        filled_mask = ndimage.binary_fill_holes(binary_mask)
    else:
        # Invert mask to identify holes
        inverted = ~binary_mask
        
        # Label holes
        labeled_holes, num_holes = ndimage.label(inverted)
        
        # Get background label (largest component)
        background_label = 0
        if num_holes > 0:
            # The background might not have label 0 if there are isolated foreground regions
            # Find the largest component which is likely the background
            component_sizes = ndimage.sum(inverted, labeled_holes, range(1, num_holes + 1))
            if len(component_sizes) > 0:
                largest_component = np.argmax(component_sizes) + 1
                background_label = largest_component
        
        # Identify holes (non-background components)
        hole_sizes = []
        for i in range(1, num_holes + 1):
            if i != background_label:
                size = np.sum(labeled_holes == i)
                if size <= max_hole_size:
                    hole_sizes.append((i, size))
        
        # Fill small holes
        filled_mask = binary_mask.copy()
        for label, _ in hole_sizes:
            filled_mask[labeled_holes == label] = True
    
    # Convert back to float32
    result = filled_mask.astype(np.float32)
    
    # Convert back to tensor if needed
    if is_tensor:
        return torch.from_numpy(result).to(device)
    else:
        return result


def largest_connected_component(
    mask: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Keep only the largest connected component.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only the largest component
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        device = mask.device
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    
    # Handle multi-channel input
    if mask_np.ndim >= 4:
        # Process each channel separately
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                mask_np[b, c] = largest_connected_component(mask_np[b, c])
        
        # Convert back to tensor if needed
        if is_tensor:
            return torch.from_numpy(mask_np).to(device)
        else:
            return mask_np
    
    # Convert to binary
    binary_mask = mask_np > 0.5
    
    # Find connected components
    labeled, num_components = ndimage.label(binary_mask)
    
    if num_components == 0:
        # No components found
        return mask
    
    # Measure component sizes
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_components + 1))
    
    # Find largest component
    largest_component = np.argmax(component_sizes) + 1
    
    # Create mask with only largest component
    result = (labeled == largest_component).astype(np.float32)
    
    # Convert back to tensor if needed
    if is_tensor:
        return torch.from_numpy(result).to(device)
    else:
        return result


def apply_postprocessing(
    pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    remove_small_objects: bool = True,
    min_size: int = 100,
    fill_holes: bool = True,
    max_hole_size: int = 100,
    largest_cc_only: bool = False,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply a series of postprocessing operations.
    
    Args:
        pred: Prediction array
        threshold: Threshold value
        remove_small_objects: Whether to remove small objects
        min_size: Minimum component size
        fill_holes: Whether to fill holes
        max_hole_size: Maximum hole size to fill
        largest_cc_only: Whether to keep only the largest component
        **kwargs: Additional parameters
        
    Returns:
        Postprocessed prediction
    """
    # Apply threshold
    binary_pred = apply_threshold(pred, threshold)
    
    # Remove small objects if requested
    if remove_small_objects:
        binary_pred = globals()["remove_small_objects"](binary_pred, min_size=min_size)
    
    # Fill holes if requested
    if fill_holes:
        binary_pred = globals()["fill_holes"](binary_pred, max_hole_size=max_hole_size)
    
    # Keep only largest component if requested
    if largest_cc_only:
        binary_pred = largest_connected_component(binary_pred)
    
    return binary_pred