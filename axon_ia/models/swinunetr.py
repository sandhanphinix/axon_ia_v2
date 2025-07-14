"""
SwinUNETR model implementation.

This module provides an implementation of the SwinUNETR architecture,
which uses Swin Transformer blocks for medical image segmentation.

Reference:
    Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images"
    https://arxiv.org/abs/2201.01266
"""

from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
from axon_ia.utils.logger import get_logger

logger = get_logger()


class SwinUNETR(nn.Module):
    """
    SwinUNETR model for medical image segmentation.
    
    This model uses Swin Transformer blocks as the encoder
    and a CNN decoder with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        use_deep_supervision: bool = False,
        # img_size and other unused kwargs for compatibility
        **kwargs
    ):
        """
        Initialize SwinUNETR model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            feature_size: Feature size for the model
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate
            use_checkpoint: Whether to use checkpointing
            use_deep_supervision: Whether to use deep supervision
        """
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        
        # Create base SwinUNETR model from MONAI
        self.swin_unetr = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )
        
        # Add deep supervision heads if requested
        # TODO: Implement deep supervision properly with MONAI 1.5.0+
        # For now, disable deep supervision to avoid compatibility issues
        if use_deep_supervision:
            logger.warning("Deep supervision temporarily disabled for MONAI compatibility")
            use_deep_supervision = False
        
        self.use_deep_supervision = use_deep_supervision
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SwinUNETR model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Use the standard MONAI SwinUNETR forward method
        return self.swin_unetr(x)