"""
Combination of multiple loss functions for medical image segmentation.

Combo Loss combines Dice Loss and Focal Loss for better performance on both
large and small structures in medical images.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union

from axon_ia.losses.dice import DiceLoss
from axon_ia.losses.focal import FocalLoss


class ComboLoss(nn.Module):
    """
    Combo Loss: combines Dice Loss and Focal Loss for better segmentation performance.
    
    This loss function is especially effective for imbalanced datasets
    where the target regions (lesions) are much smaller than the background.
    
    Reference:
        "Combo loss: Handling input and output imbalance in multi-organ segmentation"
        (https://arxiv.org/abs/1805.02798)
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5,
        squared_pred: bool = False,
        reduction: str = "mean",
        include_background: bool = False,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Initialize Combo Loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            focal_gamma: Focusing parameter for Focal loss
            smooth: Smoothing factor for Dice loss
            squared_pred: Whether to square predictions in Dice loss denominator
            reduction: Reduction method ("mean", "sum", "none")
            include_background: Whether to include background in loss calculation
            class_weights: Optional class weights for both loss components
        """
        super(ComboLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss(
            smooth=smooth, 
            squared_pred=squared_pred, 
            reduction=reduction,
            include_background=include_background
        )
        
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, 
            alpha=class_weights[1] if class_weights is not None and len(class_weights) > 1 else None,
            reduction=reduction
        )
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor
            
        Returns:
            Combined loss
        """
        # Calculate individual losses
        dice = self.dice_loss(input, target)
        focal = self.focal_loss(input, target)
        
        # Combine losses
        combo_loss = self.dice_weight * dice + self.focal_weight * focal
        
        return combo_loss