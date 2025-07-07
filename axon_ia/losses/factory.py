"""
Factory module for creating loss functions from configuration.

This module provides functions to create and configure
various loss functions based on configuration parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List

from axon_ia.losses.dice import DiceLoss, DiceCELoss
from axon_ia.losses.focal import FocalLoss
from axon_ia.losses.combo import ComboLoss
from axon_ia.losses.boundary import BoundaryLoss


def create_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Create a loss function based on type and parameters.
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional parameters for the loss function
        
    Returns:
        Instantiated loss function
    """
    loss_type = loss_type.lower()
    
    if loss_type == "dice":
        return DiceLoss(
            smooth=kwargs.get("smooth", 1e-5),
            squared_pred=kwargs.get("squared_pred", False),
            reduction=kwargs.get("reduction", "mean"),
            include_background=kwargs.get("include_background", False),
        )
    
    elif loss_type == "dice_ce" or loss_type == "dicece":
        return DiceCELoss(
            smooth=kwargs.get("smooth", 1e-5),
            squared_pred=kwargs.get("squared_pred", False),
            reduction=kwargs.get("reduction", "mean"),
            include_background=kwargs.get("include_background", False),
            ce_weight=kwargs.get("ce_weight", 1.0),
            dice_weight=kwargs.get("dice_weight", 1.0),
            class_weights=kwargs.get("class_weights", None),
        )
    
    elif loss_type == "focal":
        return FocalLoss(
            gamma=kwargs.get("gamma", 2.0),
            alpha=kwargs.get("alpha", None),
            reduction=kwargs.get("reduction", "mean"),
        )
    
    elif loss_type == "combo":
        return ComboLoss(
            dice_weight=kwargs.get("dice_weight", 1.0),
            focal_weight=kwargs.get("focal_weight", 1.0),
            focal_gamma=kwargs.get("focal_gamma", 2.0),
            smooth=kwargs.get("smooth", 1e-5),
            squared_pred=kwargs.get("squared_pred", False),
            reduction=kwargs.get("reduction", "mean"),
            include_background=kwargs.get("include_background", False),
            class_weights=kwargs.get("class_weights", None),
        )
    
    elif loss_type == "boundary":
        return BoundaryLoss(
            kernel_size=kwargs.get("kernel_size", 3),
            weight=kwargs.get("weight", 1.0),
            dice_weight=kwargs.get("dice_weight", 1.0),
            reduction=kwargs.get("reduction", "mean"),
            use_dice=kwargs.get("use_dice", True),
        )
    
    else:
        available_losses = ["dice", "dice_ce", "focal", "combo", "boundary"]
        raise ValueError(f"Unknown loss type: {loss_type}. Available losses: {available_losses}")