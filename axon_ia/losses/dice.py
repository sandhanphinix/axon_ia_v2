"""
Dice loss implementations for medical image segmentation.

This module provides Dice-based loss functions which are well-suited
for segmentation tasks with class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    
    This loss is well-suited for imbalanced segmentation problems, as it
    focuses on the overlap between prediction and ground truth rather than
    per-pixel accuracy.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        squared_pred: bool = False,
        reduction: str = "mean",
        include_background: bool = False,
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            squared_pred: Whether to square predictions in the denominator
            reduction: Reduction method ("mean", "sum", "none")
            include_background: Whether to include background class in loss calculation
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.reduction = reduction
        self.include_background = include_background
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor, either of shape:
                    - [B, C, ...] (one-hot)
                    - [B, ...] (class indices)
            weight: Optional class weights of shape [C]
            
        Returns:
            Dice loss
        """
        # Binary case handling (1 output channel)
        if input.size(1) == 1:
            # For binary segmentation, just add channel dimension to target if needed
            if target.dim() != input.dim():
                target = target.unsqueeze(1)  # Add channel dimension
            
            input = torch.sigmoid(input)
            
            # Flatten
            input = input.view(-1)
            target = target.view(-1).float()
            
            # Calculate Dice
            intersection = torch.sum(input * target)
            
            if self.squared_pred:
                denominator = torch.sum(input * input) + torch.sum(target * target)
            else:
                denominator = torch.sum(input) + torch.sum(target)
            
            dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_loss = 1.0 - dice_score
            
            if self.reduction == "mean":
                return dice_loss
            elif self.reduction == "sum":
                return dice_loss * input.size(0)
            else:  # 'none'
                return dice_loss
        
        # Multi-class case
        else:
            # Ensure target has the right shape for multi-class
            if target.dim() != input.dim():
                target = F.one_hot(target.long(), input.size(1))
                target = target.permute(0, -1, *range(1, target.dim() - 1)).float()
            
            # Apply softmax to get probabilities
            input = F.softmax(input, dim=1)
            
            # Calculate Dice for each class
            dice_loss = 0.0
            n_classes = input.size(1)
            
            # Skip background if specified
            start_idx = 1 if not self.include_background else 0
            
            for i in range(start_idx, n_classes):
                dice_loss_i = self._binary_dice_loss(
                    input[:, i], target[:, i], self.squared_pred, self.smooth
                )
                
                # Apply class weights if provided
                if weight is not None:
                    dice_loss_i = dice_loss_i * weight[i]
                
                dice_loss += dice_loss_i
            
            # Average over classes
            dice_loss = dice_loss / (n_classes - start_idx)
            
            return dice_loss
    
    def _binary_dice_loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        squared_pred: bool,
        smooth: float
    ) -> torch.Tensor:
        """
        Calculate binary Dice loss for a single class.
        
        Args:
            input: Predictions for a single class
            target: Ground truth for a single class
            squared_pred: Whether to square predictions in denominator
            smooth: Smoothing factor
            
        Returns:
            Binary Dice loss
        """
        # Flatten
        input = input.view(-1)
        target = target.view(-1).float()
        
        # Calculate Dice
        intersection = torch.sum(input * target)
        
        if squared_pred:
            denominator = torch.sum(input * input) + torch.sum(target * target)
        else:
            denominator = torch.sum(input) + torch.sum(target)
        
        dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
        
        return 1.0 - dice_score


class DiceCELoss(nn.Module):
    """
    Combination of Dice and Cross-Entropy loss.
    
    This loss combines the advantages of Dice loss (good for imbalanced classes)
    and Cross-Entropy loss (good for fine-grained boundaries).
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        squared_pred: bool = False,
        reduction: str = "mean",
        include_background: bool = False,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Initialize Dice-CE Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            squared_pred: Whether to square predictions in the denominator
            reduction: Reduction method ("mean", "sum", "none")
            include_background: Whether to include background in loss calculation
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for dice loss
            class_weights: Optional class weights for cross-entropy loss
        """
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(smooth, squared_pred, reduction, include_background)
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined Dice-CE loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor, either of shape:
                    - [B, C, ...] (one-hot)
                    - [B, ...] (class indices)
            
        Returns:
            Combined Dice-CE loss
        """
        # Ensure class_weights is on the right device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(input.device)
        
        # Calculate Dice loss
        dice_loss = self.dice(input, target)
        
        # Calculate Cross-Entropy loss
        if input.size(1) == 1:
            # Binary case
            ce_loss = F.binary_cross_entropy_with_logits(
                input, target.float(), reduction=self.reduction
            )
        else:
            # Multi-class case
            if target.dim() != input.dim():
                # Target is class indices
                ce_loss = F.cross_entropy(
                    input, target.long(), weight=self.class_weights, 
                    reduction=self.reduction
                )
            else:
                # Target is one-hot
                log_softmax = F.log_softmax(input, dim=1)
                ce_loss = -torch.sum(target * log_softmax, dim=1)
                
                if self.reduction == "mean":
                    ce_loss = ce_loss.mean()
                elif self.reduction == "sum":
                    ce_loss = ce_loss.sum()
        
        # Combine losses
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return combined_loss