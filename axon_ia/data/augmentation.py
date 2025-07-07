"""
Data augmentation for medical images.

This module provides specialized augmentation transformations
for medical imaging data, particularly 3D volumetric data.
"""

import random
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from scipy.ndimage import rotate, shift, zoom, gaussian_filter

from axon_ia.utils.logger import get_logger

logger = get_logger()


class SpatialTransformer:
    """
    Spatial transformation for 3D medical images.
    
    This class provides various spatial transformations like rotation,
    scaling, flipping, and elastic deformation.
    """
    
    def __init__(
        self,
        rotation_range: Union[float, Tuple[float, float, float]] = 10.0,
        scale_range: Union[float, Tuple[float, float]] = (0.9, 1.1),
        flip_axes: Optional[List[int]] = [0, 1, 2],
        flip_prob: float = 0.5,
        shift_range: Union[float, Tuple[float, float, float]] = 10.0,
        elastic_prob: float = 0.2,
        elastic_alpha: float = 15.0,
        elastic_sigma: float = 3.0,
        fill_value: float = 0.0,
        p: float = 1.0,
    ):
        """
        Initialize the spatial transformer.
        
        Args:
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            flip_axes: Axes to flip (0=z, 1=y, 2=x)
            flip_prob: Probability of flipping each axis
            shift_range: Range of shifts in voxels
            elastic_prob: Probability of applying elastic deformation
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            fill_value: Fill value for regions outside the input
            p: Probability of applying transformations
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_axes = flip_axes
        self.flip_prob = flip_prob
        self.shift_range = shift_range
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply spatial transformations to a sample.
        
        Args:
            sample: Dictionary with 'image' and 'mask' keys
            
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
        
        # Get image and mask
        image = sample["image"]
        mask = sample["mask"]
        
        # Convert tensors to numpy if needed
        image_is_tensor = isinstance(image, torch.Tensor)
        mask_is_tensor = isinstance(mask, torch.Tensor)
        
        if image_is_tensor:
            image = image.numpy()
        if mask_is_tensor:
            mask = mask.numpy()
        
        # Get dimensions
        c, d, h, w = image.shape
        
        # Apply random flips
        if self.flip_axes is not None:
            for axis in self.flip_axes:
                if random.random() < self.flip_prob:
                    # +1 to axis because channel is first dimension
                    image = np.flip(image, axis=axis+1)
                    mask = np.flip(mask, axis=axis)
        
        # Apply random rotations
        if isinstance(self.rotation_range, (int, float)):
            rotation_range = [-self.rotation_range, self.rotation_range]
        else:
            rotation_range = self.rotation_range
        
        if rotation_range[1] > rotation_range[0]:
            # Random rotation angles for each axis
            angles = [
                random.uniform(-rotation_range[0], rotation_range[0]),
                random.uniform(-rotation_range[0], rotation_range[0]),
                random.uniform(-rotation_range[0], rotation_range[0])
            ]
            
            # Apply rotation to each channel
            for i in range(c):
                for axis, angle in enumerate([0, 1, 2]):
                    if angle != 0:
                        # Rotate channel
                        image[i] = rotate(
                            image[i],
                            angle,
                            axes=((axis+1) % 3, (axis+2) % 3),
                            reshape=False,
                            mode='constant',
                            cval=self.fill_value
                        )
            
            # Apply same rotation to mask
            for axis, angle in enumerate([0, 1, 2]):
                if angle != 0:
                    mask = rotate(
                        mask,
                        angle,
                        axes=((axis+1) % 3, (axis+2) % 3),
                        reshape=False,
                        mode='constant',
                        cval=0
                    )
        
        # Apply random scaling
        if isinstance(self.scale_range, (int, float)):
            scale_range = [1.0 - self.scale_range, 1.0 + self.scale_range]
        else:
            scale_range = self.scale_range
        
        if scale_range[1] > scale_range[0]:
            scale_factor = random.uniform(scale_range[0], scale_range[1])
            
            # Apply scaling to each channel
            for i in range(c):
                image[i] = zoom(
                    image[i],
                    scale_factor,
                    mode='constant',
                    cval=self.fill_value
                )
            
            # Apply same scaling to mask
            mask = zoom(
                mask,
                scale_factor,
                mode='constant',
                cval=0
            )
        
        # Apply random shift
        if isinstance(self.shift_range, (int, float)):
            shift_range = [-self.shift_range, self.shift_range]
        else:
            shift_range = self.shift_range
        
        if shift_range[1] > shift_range[0]:
            # Random shift for each axis
            shifts = [
                random.uniform(shift_range[0], shift_range[1]),
                random.uniform(shift_range[0], shift_range[1]),
                random.uniform(shift_range[0], shift_range[1])
            ]
            
            # Apply shift to each channel
            for i in range(c):
                image[i] = shift(
                    image[i],
                    shifts,
                    mode='constant',
                    cval=self.fill_value
                )
            
            # Apply same shift to mask
            mask = shift(
                mask,
                shifts,
                mode='constant',
                cval=0
            )
        
        # Apply elastic deformation
        if random.random() < self.elastic_prob:
            # Generate random displacement fields
            d_field = np.random.randn(3, d, h, w) * self.elastic_alpha
            
            # Smooth displacement fields
            for i in range(3):
                d_field[i] = gaussian_filter(d_field[i], self.elastic_sigma)
            
            # Create meshgrid
            z, y, x = np.meshgrid(
                np.arange(d), np.arange(h), np.arange(w),
                indexing='ij'
            )
            
            # Add displacement
            indices = [
                np.reshape(z + d_field[0], (-1, 1)),
                np.reshape(y + d_field[1], (-1, 1)),
                np.reshape(x + d_field[2], (-1, 1))
            ]
            
            # Apply elastic deformation to each channel
            for i in range(c):
                image[i] = map_coordinates(
                    image[i],
                    indices,
                    order=3,
                    mode='constant',
                    cval=self.fill_value
                ).reshape(d, h, w)
            
            # Apply same deformation to mask
            mask = map_coordinates(
                mask,
                indices,
                order=0,
                mode='constant',
                cval=0
            ).reshape(d, h, w)
        
        # Ensure mask remains binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert back to tensors if needed
        if image_is_tensor:
            image = torch.from_numpy(image)
        if mask_is_tensor:
            mask = torch.from_numpy(mask)
        
        # Update sample
        sample["image"] = image
        sample["mask"] = mask
        
        return sample


class IntensityTransformer:
    """
    Intensity transformation for medical images.
    
    This class provides various intensity transformations like
    noise addition, gamma correction, and contrast adjustment.
    """
    
    def __init__(
        self,
        noise_prob: float = 0.2,
        noise_std: Union[float, Tuple[float, float]] = (0.01, 0.05),
        bias_prob: float = 0.3,
        bias_field_std: float = 0.3,
        gamma_prob: float = 0.3,
        gamma_range: Tuple[float, float] = (0.7, 1.3),
        contrast_prob: float = 0.2,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 1.0,
    ):
        """
        Initialize the intensity transformer.
        
        Args:
            noise_prob: Probability of adding noise
            noise_std: Standard deviation range for noise
            bias_prob: Probability of adding bias field
            bias_field_std: Standard deviation for bias field
            gamma_prob: Probability of gamma correction
            gamma_range: Range of gamma values
            contrast_prob: Probability of contrast adjustment
            contrast_range: Range of contrast factors
            p: Probability of applying transformations
        """
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.bias_prob = bias_prob
        self.bias_field_std = bias_field_std
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply intensity transformations to a sample.
        
        Args:
            sample: Dictionary with 'image' and 'mask' keys
            
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
        
        # Get image
        image = sample["image"]
        
        # Convert tensor to numpy if needed
        image_is_tensor = isinstance(image, torch.Tensor)
        if image_is_tensor:
            image = image.numpy()
        
        # Get dimensions
        c, d, h, w = image.shape
        
        # Apply random noise
        if random.random() < self.noise_prob:
            if isinstance(self.noise_std, (int, float)):
                noise_std = self.noise_std
            else:
                noise_std = random.uniform(self.noise_std[0], self.noise_std[1])
            
            for i in range(c):
                noise = np.random.normal(0, noise_std, (d, h, w))
                image[i] = image[i] + noise
        
        # Apply random bias field
        if random.random() < self.bias_prob:
            # Generate smooth bias field
            bias_field = np.random.normal(0, 1, (3, 3, 3)) * self.bias_field_std
            bias_field = zoom(bias_field, (d/3, h/3, w/3), order=3)
            
            # Apply exponential to create multiplicative field
            bias_field = np.exp(bias_field)
            
            for i in range(c):
                image[i] = image[i] * bias_field
        
        # Apply random gamma correction
        if random.random() < self.gamma_prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            
            for i in range(c):
                # Normalize to [0, 1]
                min_val = np.min(image[i])
                max_val = np.max(image[i])
                
                if max_val > min_val:
                    normalized = (image[i] - min_val) / (max_val - min_val)
                    
                    # Apply gamma
                    gamma_corrected = np.power(normalized, gamma)
                    
                    # Restore original range
                    image[i] = gamma_corrected * (max_val - min_val) + min_val
        
        # Apply random contrast adjustment
        if random.random() < self.contrast_prob:
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            
            for i in range(c):
                mean_val = np.mean(image[i])
                image[i] = (image[i] - mean_val) * contrast + mean_val
        
        # Ensure values are in reasonable range
        for i in range(c):
            p1, p99 = np.percentile(image[i], (1, 99))
            image[i] = np.clip(image[i], p1, p99)
        
        # Convert back to tensor if needed
        if image_is_tensor:
            image = torch.from_numpy(image)
        
        # Update sample
        sample["image"] = image
        
        return sample


def map_coordinates(input, indices, order, mode, cval):
    """Helper function for elastic deformation."""
    from scipy.ndimage import map_coordinates as scipy_map_coordinates
    return scipy_map_coordinates(input, indices, order=order, mode=mode, cval=cval)


class ComposeTransforms:
    """Compose multiple transformations."""
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize composition of transforms.
        
        Args:
            transforms: List of transformations to apply
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Apply all transformations in sequence.
        
        Args:
            sample: Input sample dictionary
            
        Returns:
            Transformed sample
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample