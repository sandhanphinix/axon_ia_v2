"""
UNETR (UNEt TRansformer) model implementation.

This module provides an implementation of the UNETR architecture,
which combines Transformer encoders with a CNN decoder for
medical image segmentation.

Reference:
    Hatamizadeh et al. "UNETR: Transformers for 3D Medical Image Segmentation"
    https://arxiv.org/abs/2103.10504
"""

from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT


class UNETR(nn.Module):
    """
    UNETR model for medical image segmentation.
    
    This model uses a ViT (Vision Transformer) backbone as encoder
    and a CNN decoder with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        use_deep_supervision: bool = False,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
    ):
        """
        Initialize UNETR model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            img_size: Input image size
            feature_size: Feature size for the model
            hidden_size: Hidden layer size in Transformer
            mlp_dim: MLP dimension in Transformer
            num_heads: Number of attention heads
            pos_embed: Position embedding type
            norm_name: Normalization layer type
            conv_block: Whether to use convolutional blocks
            res_block: Whether to use residual blocks
            dropout_rate: Dropout rate
            use_deep_supervision: Whether to use deep supervision
        """
        super().__init__()
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.use_deep_supervision = use_deep_supervision
        
        # Calculate patch size and dimensions
        self.patch_size = patch_size
        self.num_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, self.patch_size)])
        self.classification = False
        
        # Create Vision Transformer encoder
        # Ensure we have enough layers for skip connections (need layers 3, 6, 9)
        num_layers = max(12, 10)  # At least 10 layers to safely access layer 9
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed_type=pos_embed,  # Correct parameter name
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        
        # Encoder - decoder skip connections
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        # Decoder blocks
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Output block
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        
        # Deep supervision heads
        if self.use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                UnetOutBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=out_channels),
                UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels),
                UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels),
            ])
            
            # Set deep supervision weights (can be tuned)
            self.deep_supervision_weights = [0.4, 0.3, 0.2, 0.1]
    
    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the UNETR model.
        
        Args:
            x_in: Input tensor
            
        Returns:
            Output tensor or tuple of output tensors (if deep supervision is used)
        """
        # Transformer encoder
        x_output, hidden_states = self.vit(x_in)
        
        # Reshape hidden states from (B, num_patches, hidden_size) to (B, hidden_size, D, H, W)
        batch_size = x_in.shape[0]
        # Calculate spatial dimensions after patching
        patch_dims = [dim // patch_dim for dim, patch_dim in zip(self.img_size, self.patch_size)]
        
        def reshape_hidden_state(hidden_state):
            """Reshape from (B, num_patches, hidden_size) to (B, hidden_size, D, H, W)"""
            return hidden_state.transpose(1, 2).view(
                batch_size, self.hidden_size, patch_dims[0], patch_dims[1], patch_dims[2]
            )
        
        # Extract and reshape intermediate features 
        enc1 = self.encoder1(x_in)
        # Ensure we don't exceed the available hidden states
        max_layer = len(hidden_states) - 1
        layer_3_idx = min(3, max_layer)
        layer_6_idx = min(6, max_layer)
        layer_9_idx = min(9, max_layer)
        
        enc2 = self.encoder2(reshape_hidden_state(hidden_states[layer_3_idx]))
        enc3 = self.encoder3(reshape_hidden_state(hidden_states[layer_6_idx]))
        enc4 = self.encoder4(reshape_hidden_state(hidden_states[layer_9_idx]))
        
        # Reshape final output
        x_output_reshaped = reshape_hidden_state(x_output)
        
        # Decoder path
        dec5 = self.decoder5(x_output_reshaped, enc4)
        dec4 = self.decoder4(dec5, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)
        
        logits = self.out(dec2)
        
        # Return intermediate outputs for deep supervision if enabled
        if self.use_deep_supervision:
            # Generate predictions at different scales
            deep_outputs = [
                logits,
                self.deep_supervision_heads[0](dec5),
                self.deep_supervision_heads[1](dec4),
                self.deep_supervision_heads[2](dec3),
            ]
            return tuple(deep_outputs)
        else:
            return logits