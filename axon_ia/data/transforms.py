"""
Data transformation pipelines for medical images.

This module provides transformation pipelines for preprocessing
and augmenting medical images during training and inference.
"""

from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImage, Orientation, Spacing, Resize,
    RandRotate, RandFlip, RandZoom, RandGaussianNoise,
    NormalizeIntensity, ScaleIntensity, CropForeground,
    SpatialPad, RandSpatialCrop, ToTensor
)

from axon_ia.data.augmentation import (
    SpatialTransformer,
    IntensityTransformer,
    ComposeTransforms
)
from axon_ia.data.preprocessing import (
    normalize_intensity,
    crop_foreground,
    brain_extraction
)
from axon_ia.utils.logger import get_logger

logger = get_logger()


class NormalizeImage:
    """Normalize image intensities."""
    
    def __init__(
        self,
        mode: str = 'z_score',
        percentiles: Tuple[float, float] = (1, 99),
        channel_wise: bool = True,
        mask_key: Optional[str] = None,
    ):
        """
        Initialize intensity normalization.
        
        Args:
            mode: Normalization mode ('z_score', 'percentile', 'min_max')
            percentiles: Percentiles for clipping in percentile mode
            channel_wise: Whether to normalize each channel separately
            mask_key: Key to mask in sample dict (if None, no masking)
        """
        self.mode = mode
        self.percentiles = percentiles
        self.channel_wise = channel_wise
        self.mask_key = mask_key
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply normalization to sample."""
        image = sample["image"]
        mask = sample.get(self.mask_key, None) if self.mask_key else None
        
        # Convert tensor to numpy if needed
        image_is_tensor = isinstance(image, torch.Tensor)
        if image_is_tensor:
            image = image.numpy()
        
        # Get dimensions
        if image.ndim == 4:  # [C, D, H, W]
            c = image.shape[0]
            # Normalize each channel separately if requested
            if self.channel_wise:
                for i in range(c):
                    image[i] = normalize_intensity(
                        image[i],
                        mode=self.mode,
                        mask=mask,
                        percentiles=self.percentiles
                    )
            else:
                image = normalize_intensity(
                    image,
                    mode=self.mode,
                    mask=mask,
                    percentiles=self.percentiles
                )
        else:  # [D, H, W]
            image = normalize_intensity(
                image,
                mode=self.mode,
                mask=mask,
                percentiles=self.percentiles
            )
        
        # Convert back to tensor if needed
        if image_is_tensor:
            image = torch.from_numpy(image)
        
        sample["image"] = image
        return sample


class CropForegroundd:
    """Crop image and mask to focus on foreground region."""
    
    def __init__(
        self,
        keys: List[str],
        source_key: str = "image",
        threshold: float = 0.01,
        margin: int = 10,
    ):
        """
        Initialize foreground cropping.
        
        Args:
            keys: Keys to crop in sample dict
            source_key: Key to use for determining foreground
            threshold: Threshold for foreground in source image
            margin: Margin around foreground in voxels
        """
        self.keys = keys
        self.source_key = source_key
        self.threshold = threshold
        self.margin = margin
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply cropping to sample."""
        # Get source data for determining foreground
        source = sample[self.source_key]
        
        # Convert tensor to numpy if needed
        conversions = {}
        for key in self.keys:
            if key in sample:
                data = sample[key]
                if isinstance(data, torch.Tensor):
                    conversions[key] = True
                    sample[key] = data.numpy()
                else:
                    conversions[key] = False
        
        # Determine foreground from source
        if isinstance(source, torch.Tensor):
            source = source.numpy()
        
        # Handle multi-channel source
        if source.ndim == 4:  # [C, D, H, W]
            # Use maximum across channels
            foreground = np.max(source, axis=0) > self.threshold
        else:  # [D, H, W]
            foreground = source > self.threshold
        
        # Find bounding box
        if not np.any(foreground):
            return sample
        
        # Get indices of non-zero elements
        indices = np.where(foreground)
        
        # Get min/max indices with margin
        mins = [max(0, int(np.min(idx)) - self.margin) for idx in indices]
        maxs = [min(s, int(np.max(idx)) + self.margin + 1) for s, idx in zip(foreground.shape, indices)]
        
        # Create crop slices
        crop_slices = tuple(slice(min_val, max_val) for min_val, max_val in zip(mins, maxs))
        
        # Apply cropping to each key
        for key in self.keys:
            if key in sample:
                data = sample[key]
                
                # Apply cropping
                if data.ndim == 4:  # [C, D, H, W]
                    # Include channel dimension in crop
                    crop_slices_data = (slice(None),) + crop_slices
                    sample[key] = data[crop_slices_data]
                else:  # [D, H, W]
                    sample[key] = data[crop_slices]
        
        # Convert back to tensors if needed
        for key, was_tensor in conversions.items():
            if was_tensor and key in sample:
                sample[key] = torch.from_numpy(sample[key])
        
        return sample


class BrainExtraction:
    """Extract brain region from MRI image."""
    
    def __init__(
        self,
        apply_to_keys: List[str] = ["image"],
        mask_key: str = "brain_mask",
        method: str = 'threshold',
        threshold: float = 0.01,
        closing_iterations: int = 5,
        apply_mask: bool = True,
    ):
        """
        Initialize brain extraction.
        
        Args:
            apply_to_keys: Keys to apply brain masking to
            mask_key: Key to store brain mask in
            method: Extraction method ('threshold', 'otsu')
            threshold: Threshold value for 'threshold' method
            closing_iterations: Number of closing iterations
            apply_mask: Whether to apply the mask to the images
        """
        self.apply_to_keys = apply_to_keys
        self.mask_key = mask_key
        self.method = method
        self.threshold = threshold
        self.closing_iterations = closing_iterations
        self.apply_mask = apply_mask
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply brain extraction to sample."""
        # Get first key to extract brain mask from
        if len(self.apply_to_keys) == 0 or self.apply_to_keys[0] not in sample:
            return sample
        
        source_key = self.apply_to_keys[0]
        source = sample[source_key]
        
        # Convert tensor to numpy if needed
        source_is_tensor = isinstance(source, torch.Tensor)
        if source_is_tensor:
            source = source.numpy()
        
        # Extract brain mask
        brain_mask = brain_extraction(
            source,
            method=self.method,
            threshold=self.threshold,
            closing_iterations=self.closing_iterations
        )
        
        # Store brain mask in sample
        if self.mask_key:
            sample[self.mask_key] = brain_mask
        
        # Apply mask to specified keys if requested
        if self.apply_mask:
            for key in self.apply_to_keys:
                if key in sample:
                    data = sample[key]
                    
                    # Convert to numpy if needed
                    data_is_tensor = isinstance(data, torch.Tensor)
                    if data_is_tensor:
                        data = data.numpy()
                    
                    # Apply mask
                    if data.ndim == 4:  # [C, D, H, W]
                        for i in range(data.shape[0]):
                            data[i] = data[i] * brain_mask
                    else:  # [D, H, W]
                        data = data * brain_mask
                    
                    # Convert back to tensor if needed
                    if data_is_tensor:
                        data = torch.from_numpy(data)
                    
                    sample[key] = data
        
        return sample


class ToTensord:
    """Convert numpy arrays to tensors."""
    
    def __init__(self, keys: List[str]):
        """
        Initialize tensor conversion.
        
        Args:
            keys: Keys to convert to tensors
        """
        self.keys = keys
    
    def __call__(self, sample: Dict) -> Dict:
        """Convert arrays to tensors."""
        for key in self.keys:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])
        
        return sample


def get_default_transform() -> Callable:
    """Get default transform pipeline."""
    return ComposeTransforms([
        NormalizeImage(mode='z_score', channel_wise=True),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        ToTensord(keys=["image", "mask"])
    ])


def get_train_transform(
    rotation_range: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    flip_prob: float = 0.5,
    noise_prob: float = 0.2,
    noise_std: float = 0.03,
    gamma_prob: float = 0.3,
    gamma_range: Tuple[float, float] = (0.7, 1.3),
) -> Callable:
    """
    Get training transform pipeline with augmentations.
    
    Args:
        rotation_range: Range of rotation angles in degrees
        scale_range: Range of scaling factors
        flip_prob: Probability of flipping
        noise_prob: Probability of adding noise
        noise_std: Standard deviation of noise
        gamma_prob: Probability of gamma correction
        gamma_range: Range of gamma values
        
    Returns:
        Composed transform function
    """
    return ComposeTransforms([
        NormalizeImage(mode='z_score', channel_wise=True),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        SpatialTransformer(
            rotation_range=rotation_range,
            scale_range=scale_range,
            flip_prob=flip_prob,
        ),
        IntensityTransformer(
            noise_prob=noise_prob,
            noise_std=noise_std,
            gamma_prob=gamma_prob,
            gamma_range=gamma_range,
        ),
        ToTensord(keys=["image", "mask"])
    ])


def get_val_transform() -> Callable:
    """Get validation transform pipeline."""
    return get_default_transform()


def get_test_transform() -> Callable:
    """Get test transform pipeline."""
    return get_default_transform()