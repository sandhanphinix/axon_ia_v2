"""
Sliding window inference for large medical volumes.

This module provides memory-efficient sliding window inference
for processing large 3D volumes with limited GPU memory.
"""

import time
from typing import Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np

from axon_ia.utils.logger import get_logger

logger = get_logger()


class SlidingWindowInference:
    """
    Memory-efficient sliding window inference for 3D volumes.
    
    This class handles the process of dividing large volumes into
    overlapping patches, running inference on each patch, and
    combining the results to form a complete prediction.
    """
    
    def __init__(
        self,
        roi_size: Union[int, Tuple[int, int, int]] = 128,
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = "gaussian",
        padding_mode: str = "constant",
        device: Optional[torch.device] = None,
        use_test_time_augmentation: bool = False,
    ):
        """
        Initialize sliding window inference.
        
        Args:
            roi_size: Size of sliding window ROI (patch size)
            sw_batch_size: Batch size for sliding window inference
            overlap: Amount of overlap between adjacent windows
            mode: Blending mode for overlapping windows
            padding_mode: Padding mode for input volume
            device: Device to use for inference
            use_test_time_augmentation: Whether to use test-time augmentation
        """
        self.roi_size = roi_size if isinstance(roi_size, tuple) else (roi_size, roi_size, roi_size)
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.use_test_time_augmentation = use_test_time_augmentation
    
    def __call__(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Run sliding window inference on input tensor.
        
        Args:
            model: Neural network model
            inputs: Input tensor [B, C, D, H, W]
            
        Returns:
            Prediction tensor [B, C, D, H, W]
        """
        # Get dimensions
        batch_size, n_channels = inputs.shape[:2]
        spatial_size = inputs.shape[2:]
        num_spatial_dims = len(spatial_size)
        
        if num_spatial_dims != 3:
            raise ValueError(f"Expected 3D input, got {num_spatial_dims}D")
        
        # Set device
        if self.device is None:
            self.device = next(model.parameters()).device
        
        # Move inputs to device if needed
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        
        # Calculate window centers
        centers = self._calculate_window_centers(spatial_size)
        
        # Create importance map
        importance_map = self._create_importance_map()
        
        # Initialize output tensor
        outputs = torch.zeros(
            (batch_size, model.out_channels if hasattr(model, 'out_channels') else 1, *spatial_size),
            device=self.device
        )
        count_map = torch.zeros_like(outputs)
        
        # Process each batch
        for batch_idx in range(batch_size):
            # Process window centers in batches
            for start_idx in range(0, len(centers), self.sw_batch_size):
                end_idx = min(start_idx + self.sw_batch_size, len(centers))
                windows_batch = []
                windows_locations = []
                
                # Extract windows for this batch
                for center in centers[start_idx:end_idx]:
                    # Calculate window bounds
                    window_location = []
                    for i, (center_i, roi_i, size_i) in enumerate(zip(center, self.roi_size, spatial_size)):
                        start = max(0, center_i - roi_i // 2)
                        end = min(size_i, center_i + roi_i // 2 + roi_i % 2)
                        window_location.append((start, end))
                    
                    # Extract window and pad if needed
                    window = inputs[batch_idx:batch_idx+1]
                    for i, (start, end) in enumerate(window_location):
                        # Handle padding
                        pad_before = max(0, roi_i // 2 - center_i)
                        pad_after = max(0, center_i + roi_i // 2 + roi_i % 2 - size_i)
                        
                        if pad_before > 0 or pad_after > 0:
                            # Pad the dimension
                            pad = [0, 0] * num_spatial_dims
                            pad[i*2] = pad_before
                            pad[i*2+1] = pad_after
                            window = torch.nn.functional.pad(window, tuple(pad), mode=self.padding_mode)
                        
                        # Slice the dimension
                        dim_slice = slice(start + pad_before, end + pad_before)
                        window = window.narrow(i + 2, dim_slice.start, dim_slice.stop - dim_slice.start)
                    
                    windows_batch.append(window)
                    windows_locations.append(window_location)
                
                # Stack windows to form batch
                windows_batch = torch.cat(windows_batch, dim=0)
                
                # Run inference
                with torch.no_grad():
                    if self.use_test_time_augmentation:
                        # Apply test-time augmentation
                        window_preds = self._test_time_augmentation(model, windows_batch)
                    else:
                        # Standard forward pass
                        window_preds = model(windows_batch)
                        
                        # If model returns tuple (e.g., with deep supervision), take the first output
                        if isinstance(window_preds, (tuple, list)):
                            window_preds = window_preds[0]
                
                # Process results for each window
                for idx, (window_pred, window_location) in enumerate(zip(window_preds, windows_locations)):
                    # Get slice ranges
                    slices = tuple(slice(start, end) for start, end in window_location)
                    
                    # Get importance map for this window
                    window_importance = importance_map.to(window_pred.device)
                    
                    # Update output and count map
                    outputs[batch_idx, :, slices[0], slices[1], slices[2]] += window_pred * window_importance
                    count_map[batch_idx, :, slices[0], slices[1], slices[2]] += window_importance
        
        # Normalize by count map (avoid division by zero)
        outputs = outputs / (count_map + 1e-8)
        
        return outputs
    
    def _calculate_window_centers(self, spatial_size: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Calculate window centers for sliding window inference.
        
        Args:
            spatial_size: Spatial dimensions of input volume
            
        Returns:
            List of window center coordinates
        """
        # Calculate step size based on overlap
        step_size = [int(roi_i * (1 - self.overlap)) for roi_i in self.roi_size]
        
        # Ensure step size is at least 1
        step_size = [max(1, step) for step in step_size]
        
        # Calculate window centers
        centers = []
        for z in range(self.roi_size[0] // 2, spatial_size[0], step_size[0]):
            for y in range(self.roi_size[1] // 2, spatial_size[1], step_size[1]):
                for x in range(self.roi_size[2] // 2, spatial_size[2], step_size[2]):
                    # Add final window at the end of each dimension if needed
                    center_z = min(z, spatial_size[0] - self.roi_size[0] // 2)
                    center_y = min(y, spatial_size[1] - self.roi_size[1] // 2)
                    center_x = min(x, spatial_size[2] - self.roi_size[2] // 2)
                    centers.append((center_z, center_y, center_x))
        
        return centers
    
    def _create_importance_map(self) -> torch.Tensor:
        """
        Create importance map for blending overlapping windows.
        
        Returns:
            Importance map tensor
        """
        # Create base tensor of ROI size
        importance = torch.ones(self.roi_size)
        
        if self.mode == "constant":
            # Uniform weighting
            return importance
        
        elif self.mode == "gaussian":
            # Gaussian weighting
            # Calculate coordinates relative to center
            zs = torch.linspace(-1, 1, self.roi_size[0])
            ys = torch.linspace(-1, 1, self.roi_size[1])
            xs = torch.linspace(-1, 1, self.roi_size[2])
            
            # Create coordinate grid
            z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
            
            # Calculate squared distance from center
            squared_distance = z**2 + y**2 + x**2
            
            # Apply Gaussian function
            sigma = 0.5  # Standard deviation
            importance = torch.exp(-squared_distance / (2 * sigma**2))
        
        elif self.mode == "linear":
            # Linear weighting
            # Calculate coordinates relative to center
            zs = torch.abs(torch.linspace(-1, 1, self.roi_size[0]))
            ys = torch.abs(torch.linspace(-1, 1, self.roi_size[1]))
            xs = torch.abs(torch.linspace(-1, 1, self.roi_size[2]))
            
            # Create coordinate grid
            z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
            
            # Calculate maximum coordinate in each position
            max_coord = torch.max(torch.stack([z, y, x]), dim=0)[0]
            
            # Linear falloff from center
            importance = 1.0 - max_coord
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Reshape importance map to match output dimensions (1, 1, D, H, W)
        return importance.unsqueeze(0).unsqueeze(0)
    
    def _test_time_augmentation(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply test-time augmentation during inference.
        
        Args:
            model: Neural network model
            inputs: Input tensor
            
        Returns:
            Augmented prediction
        """
        # Define augmentations
        augmentations = [
            lambda x: x,  # Identity
            lambda x: torch.flip(x, dims=[-1]),  # Flip X
            lambda x: torch.flip(x, dims=[-2]),  # Flip Y
            lambda x: torch.flip(x, dims=[-3]),  # Flip Z
            lambda x: torch.flip(x, dims=[-1, -2]),  # Flip X, Y
            lambda x: torch.flip(x, dims=[-1, -3]),  # Flip X, Z
            lambda x: torch.flip(x, dims=[-2, -3]),  # Flip Y, Z
            lambda x: torch.flip(x, dims=[-1, -2, -3]),  # Flip X, Y, Z
        ]
        
        # Define inverse augmentations
        inverse_augmentations = [
            lambda x: x,  # Identity
            lambda x: torch.flip(x, dims=[-1]),  # Flip X
            lambda x: torch.flip(x, dims=[-2]),  # Flip Y
            lambda x: torch.flip(x, dims=[-3]),  # Flip Z
            lambda x: torch.flip(x, dims=[-1, -2]),  # Flip X, Y
            lambda x: torch.flip(x, dims=[-1, -3]),  # Flip X, Z
            lambda x: torch.flip(x, dims=[-2, -3]),  # Flip Y, Z
            lambda x: torch.flip(x, dims=[-1, -2, -3]),  # Flip X, Y, Z
        ]
        
        # Apply each augmentation and aggregate results
        outputs = []
        for aug, inv_aug in zip(augmentations, inverse_augmentations):
            # Apply augmentation
            aug_inputs = aug(inputs)
            
            # Run model
            with torch.no_grad():
                aug_outputs = model(aug_inputs)
                
                # If model returns tuple (e.g., with deep supervision), take the first output
                if isinstance(aug_outputs, (tuple, list)):
                    aug_outputs = aug_outputs[0]
            
            # Apply inverse augmentation
            outputs.append(inv_aug(aug_outputs))
        
        # Average results
        return torch.stack(outputs).mean(dim=0)