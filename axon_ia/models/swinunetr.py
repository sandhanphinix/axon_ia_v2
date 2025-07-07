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


class SwinUNETR(nn.Module):
    """
    SwinUNETR model for medical image segmentation.
    
    This model uses Swin Transformer blocks as the encoder
    and a CNN decoder with skip connections.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 4,
        out_channels: int = 1,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        use_deep_supervision: bool = False,
    ):
        """
        Initialize SwinUNETR model.
        
        Args:
            img_size: Input image size
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
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )
        
        # Add deep supervision heads if requested
        if use_deep_supervision:
            # Define deep supervision heads
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(feature_size * 8, out_channels, kernel_size=1),
                nn.Conv3d(feature_size * 4, out_channels, kernel_size=1),
                nn.Conv3d(feature_size * 2, out_channels, kernel_size=1),
            ])
            
            # Define upsampling for deep supervision outputs
            self.deep_supervision_upsamples = nn.ModuleList([
                nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False),
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ])
            
            # Set deep supervision weights (can be tuned)
            self.deep_supervision_weights = [0.4, 0.3, 0.2, 0.1]
            
            # We need to modify the forward method of the base model
            # to get the intermediate features
            self._patch_forward_method()
    
    def _patch_forward_method(self):
        """Patch the forward method of the base model to get intermediate features."""
        # Store the original forward method
        original_forward = self.swin_unetr.forward
        
        # Define our patched forward method
        def patched_forward(x):
            # Get encoder features
            x_enc1, x_enc2, x_enc3, x_enc4 = self.swin_unetr.get_encoder_features(x)
            
            # Bottleneck
            x_tr = self.swin_unetr.encoder_tr(x_enc4)
            
            # Decoder path
            x5 = self.swin_unetr.decoder_5(x_tr, x_enc4)
            x4 = self.swin_unetr.decoder_4(x5, x_enc3)
            x3 = self.swin_unetr.decoder_3(x4, x_enc2)
            x2 = self.swin_unetr.decoder_2(x3, x_enc1)
            
            # Final output
            logits = self.swin_unetr.out(x2)
            
            # Store intermediate decoder features for deep supervision
            self.deep_features = [x5, x4, x3]
            
            return logits
        
        # Replace the forward method
        self.swin_unetr.forward = patched_forward
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the SwinUNETR model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor or tuple of output tensors (if deep supervision is used)
        """
        # Call the base model's forward method
        logits = self.swin_unetr(x)
        
        # Return intermediate outputs for deep supervision if enabled
        if self.use_deep_supervision and self.training:
            # Get intermediate features
            deep_outputs = [logits]
            
            # Generate predictions from intermediate features
            for i, (feature, head, upsample) in enumerate(zip(
                self.deep_features,
                self.deep_supervision_heads,
                self.deep_supervision_upsamples
            )):
                # Apply head and upsample to match original size
                deep_output = upsample(head(feature))
                deep_outputs.append(deep_output)
            
            return tuple(deep_outputs)
        else:
            return logits