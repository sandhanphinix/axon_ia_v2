"""
Boundary-aware loss functions for medical image segmentation.

This module provides loss functions that focus on accurate boundary delineation,
which is crucial for precise segmentation of lesions and anatomical structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from axon_ia.losses.dice import DiceLoss


class BoundaryLoss(nn.Module):
    """
    Boundary loss for improving segmentation accuracy at object boundaries.
    
    This loss penalizes errors at the boundaries more heavily than errors
    in the interior regions, leading to more precise delineation.
    
    Reference:
        "Boundary loss for highly unbalanced segmentation"
        (https://arxiv.org/abs/1812.07032)
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        weight: float = 1.0,
        dice_weight: float = 1.0,
        reduction: str = "mean",
        use_dice: bool = True,
    ):
        """
        Initialize Boundary Loss.
        
        Args:
            kernel_size: Size of the kernel used for boundary extraction
            weight: Weight for boundary loss component
            dice_weight: Weight for Dice loss component (if used)
            reduction: Reduction method ("mean", "sum", "none")
            use_dice: Whether to combine with Dice loss
        """
        super(BoundaryLoss, self).__init__()
        
        self.kernel_size = kernel_size
        self.weight = weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.use_dice = use_dice
        
        # Initialize Dice loss if needed
        if self.use_dice:
            self.dice_loss = DiceLoss(reduction=reduction)
        
        # Create the boundary extraction kernel
        self.laplacian_kernel = self._get_laplacian_kernel()
    
    def _get_laplacian_kernel(self) -> torch.Tensor:
        """
        Create a 3D Laplacian kernel for boundary extraction.
        
        Returns:
            Laplacian kernel tensor
        """
        # Create 3D kernel
        kernel_size = self.kernel_size
        kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size))
        
        # Set center value
        center = kernel_size // 2
        kernel[0, 0, center, center, center] = 2 * 3  # 2 * dimensions
        
        # Set neighbor values
        kernel[0, 0, center-1:center+2, center, center] = -1
        kernel[0, 0, center, center-1:center+2, center] = -1
        kernel[0, 0, center, center, center-1:center+2] = -1
        
        return kernel
    
    def _extract_boundaries(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries using a Laplacian filter.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Boundary tensor
        """
        # Ensure kernel is on the right device
        if self.laplacian_kernel.device != tensor.device:
            self.laplacian_kernel = self.laplacian_kernel.to(tensor.device)
        
        # Apply Laplacian filter
        boundaries = F.conv3d(
            tensor, self.laplacian_kernel, padding=self.kernel_size // 2
        )
        
        # Take absolute values to get boundary strength
        boundaries = torch.abs(boundaries)
        
        return boundaries
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate boundary loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor
            
        Returns:
            Boundary loss
        """
        # Apply appropriate activation based on number of classes
        if input.size(1) == 1:
            probs = torch.sigmoid(input)
        else:
            probs = F.softmax(input, dim=1)
        
        # Extract boundaries from predictions and ground truth
        if input.size(1) == 1:
            # Binary case
            pred_boundaries = self._extract_boundaries(probs)
            target_boundaries = self._extract_boundaries(target.float())
        else:
            # Multi-class case
            pred_boundaries = torch.zeros_like(probs)
            target_boundaries = torch.zeros_like(target.float())
            
            for i in range(input.size(1)):
                pred_boundaries[:, i:i+1] = self._extract_boundaries(probs[:, i:i+1])
                target_boundaries[:, i:i+1] = self._extract_boundaries(target[:, i:i+1].float())
        
        # Calculate MSE loss between boundaries
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries, reduction=self.reduction)
        
        # Combine with Dice loss if needed
        if self.use_dice:
            dice = self.dice_loss(input, target)
            return self.weight * boundary_loss + self.dice_weight * dice
        else:
            return self.weight * boundary_loss