"""
Focal loss implementation for medical image segmentation.

Focal Loss addresses class imbalance by down-weighting the loss contribution
of well-classified examples, focusing more on hard examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in segmentation.
    
    This loss focuses on hard examples by down-weighting the contribution
    of easy-to-classify examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" 
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter that controls down-weighting of well-classified examples
            alpha: Class balancing parameter (None, float, or list of floats per class)
            reduction: Reduction method ("mean", "sum", "none")
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor, either of shape:
                    - [B, C, ...] (one-hot)
                    - [B, ...] (class indices)
            
        Returns:
            Focal loss
        """
        # Binary case handling
        if input.size(1) == 1:
            return self._binary_focal_loss(input, target)
        else:
            return self._multiclass_focal_loss(input, target)
    
    def _binary_focal_loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate binary focal loss.
        
        Args:
            input: Model output tensor of shape [B, 1, ...]
            target: Ground truth tensor of shape [B, 1, ...] or [B, ...]
            
        Returns:
            Binary focal loss
        """
        # Ensure target is the right shape
        if target.dim() < input.dim():
            target = target.unsqueeze(1)
        
        # Flatten
        input = input.view(-1)
        target = target.view(-1).float()
        
        # Apply sigmoid
        probs = torch.sigmoid(input)
        pt = torch.where(target == 1, probs, 1 - probs)
        
        # Compute alpha term
        if self.alpha is not None:
            alpha = torch.tensor([self.alpha, 1 - self.alpha]).to(input.device)
            alpha_t = torch.where(target == 1, alpha[0], alpha[1])
            focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-10)
        else:
            focal_loss = -(1 - pt) ** self.gamma * torch.log(pt + 1e-10)
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
    def _multiclass_focal_loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate multi-class focal loss.
        
        Args:
            input: Model output tensor of shape [B, C, ...]
            target: Ground truth tensor of shape [B, C, ...] or [B, ...]
            
        Returns:
            Multi-class focal loss
        """
        # Get original dimensions
        n_dims = input.dim()
        
        # Convert target to one-hot if needed
        if target.dim() != input.dim():
            target = F.one_hot(target.long(), input.size(1))
            # Move class dimension to correct position
            target = target.permute(0, -1, *range(1, n_dims - 1))
        
        # Apply softmax to get probabilities
        log_softmax = F.log_softmax(input, dim=1)
        softmax = torch.exp(log_softmax)
        
        # Compute focal weights
        focal_weights = (1 - softmax) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha).to(input.device)
            else:
                alpha = torch.tensor([self.alpha, 1 - self.alpha]).to(input.device)
            
            # Ensure alpha has right shape
            if alpha.size(0) != input.size(1):
                raise ValueError(f"Alpha size ({alpha.size(0)}) must match number of classes ({input.size(1)})")
            
            # Apply alpha weighting
            focal_weights = focal_weights * alpha.view(1, -1, *([1] * (n_dims - 2)))
        
        # Compute focal loss
        focal_loss = -focal_weights * target * log_softmax
        
        # Sum over classes
        focal_loss = torch.sum(focal_loss, dim=1)
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss