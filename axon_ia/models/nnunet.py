"""
nnUNet model implementation.

This module provides an implementation of the nnUNet architecture,
which is an automatic segmentation framework that has achieved
state-of-the-art performance in various medical imaging tasks.

Reference:
    Isensee et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
    https://www.nature.com/articles/s41592-020-01008-z
"""

from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard convolution block with normalization and activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str = "instance",
        dropout: float = 0.0,
    ):
        """
        Initialize convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            norm_name: Type of normalization layer ('batch', 'instance', 'group')
            dropout: Dropout probability
        """
        super().__init__()
        
        # Determine padding based on kernel size
        padding = kernel_size // 2
        
        # Determine normalization layer
        if norm_name == "batch":
            norm_layer = nn.BatchNorm3d
        elif norm_name == "instance":
            norm_layer = nn.InstanceNorm3d
        elif norm_name == "group":
            norm_layer = lambda channels: nn.GroupNorm(num_groups=8, num_channels=channels)
        else:
            raise ValueError(f"Unknown normalization: {norm_name}")
        
        # Build the block
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=norm_name != "batch"  # No bias when using batch norm
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolution block.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with normalization and activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str = "instance",
        dropout: float = 0.0,
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            norm_name: Type of normalization layer
            dropout: Dropout probability
        """
        super().__init__()
        
        # First convolution block
        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm_name,
            dropout
        )
        
        # Second convolution block
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size,
            1,  # stride
            norm_name,
            dropout
        )
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + res
        return x


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features.
    """
    
    def __init__(
        self,
        gate_channels: int,
        feat_channels: int,
        inter_channels: Optional[int] = None,
    ):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Number of channels in the gating signal
            feat_channels: Number of channels in the feature map
            inter_channels: Number of intermediate channels
        """
        super().__init__()
        
        # Set intermediate channels if not provided
        if inter_channels is None:
            inter_channels = gate_channels // 2
        
        # Convolutional layers
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(feat_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the attention gate.
        
        Args:
            g: Gating signal tensor
            x: Feature map tensor
            
        Returns:
            Attention weighted feature map
        """
        # Process the gating signal
        g1 = self.W_g(g)
        
        # Process the feature map
        x1 = self.W_x(x)
        
        # Perform addition and activation
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights
        return x * psi


class DownsamplingBlock(nn.Module):
    """
    Downsampling block for encoder path.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        norm_name: str = "instance",
        dropout: float = 0.0,
        res_block: bool = True,
    ):
        """
        Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride for downsampling
            norm_name: Type of normalization layer
            dropout: Dropout probability
            res_block: Whether to use residual blocks
        """
        super().__init__()
        
        if res_block:
            self.down = ResidualBlock(
                in_channels, out_channels, kernel_size, stride, norm_name, dropout
            )
        else:
            self.down = ConvBlock(
                in_channels, out_channels, kernel_size, stride, norm_name, dropout
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the downsampling block.
        
        Args:
            x: Input tensor
            
        Returns:
            Downsampled tensor
        """
        return self.down(x)


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for decoder path.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        norm_name: str = "instance",
        dropout: float = 0.0,
        res_block: bool = True,
        use_attention: bool = False,
    ):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride for upsampling
            norm_name: Type of normalization layer
            dropout: Dropout probability
            res_block: Whether to use residual blocks
            use_attention: Whether to use attention gates
        """
        super().__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False
        )
        
        # Attention gate
        self.attention = AttentionGate(out_channels, out_channels) if use_attention else None
        
        # Processing block
        if res_block:
            self.conv = ResidualBlock(
                2 * out_channels, out_channels, kernel_size, 1, norm_name, dropout
            )
        else:
            self.conv = nn.Sequential(
                ConvBlock(2 * out_channels, out_channels, kernel_size, 1, norm_name, dropout),
                ConvBlock(out_channels, out_channels, kernel_size, 1, norm_name, dropout)
            )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x: Input tensor
            skip: Skip connection tensor
            
        Returns:
            Upsampled and processed tensor
        """
        # Upsample
        x = self.up(x)
        
        # Apply attention if used
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Process
        x = self.conv(x)
        
        return x


class NNUNet(nn.Module):
    """
    nnU-Net architecture for medical image segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        feature_channels: List[int] = [32, 64, 128, 256, 320, 320],
        strides: List[int] = [1, 2, 2, 2, 2, 2],
        kernel_size: int = 3,
        norm_name: str = "instance",
        dropout: float = 0.0,
        deep_supervision: bool = False,
        deep_supr_num: int = 2,
        res_block: bool = True,
        attention_gate: bool = False,
    ):
        """
        Initialize nnU-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            feature_channels: List of feature channels at each level
            strides: List of strides for downsampling
            kernel_size: Size of the convolutional kernels
            norm_name: Type of normalization layer
            dropout: Dropout probability
            deep_supervision: Whether to use deep supervision
            deep_supr_num: Number of deep supervision outputs
            res_block: Whether to use residual blocks
            attention_gate: Whether to use attention gates
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num if deep_supervision else 0
        
        # Check parameters
        if len(feature_channels) != len(strides):
            raise ValueError("Length of feature_channels and strides must match")
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i, (out_ch, stride) in enumerate(zip(feature_channels, strides)):
            self.encoders.append(
                DownsamplingBlock(
                    in_ch, out_ch, kernel_size, stride, norm_name, dropout, res_block
                )
            )
            in_ch = out_ch
        
        # Decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(len(feature_channels) - 1, 0, -1):
            # Skip 1x1 strides as they don't need upsampling
            if strides[i] > 1:
                self.decoders.append(
                    UpsamplingBlock(
                        feature_channels[i],
                        feature_channels[i - 1],
                        kernel_size,
                        strides[i],
                        norm_name,
                        dropout,
                        res_block,
                        attention_gate
                    )
                )
        
        # Output convolution
        self.output = nn.Conv3d(feature_channels[0], out_channels, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(deep_supr_num):
                ds_level = i + 1
                self.deep_supervision_heads.append(
                    nn.Conv3d(feature_channels[ds_level], out_channels, kernel_size=1)
                )
            
            # Set deep supervision weights
            self.deep_supervision_weights = [
                1.0 / (2**i) for i in range(deep_supr_num + 1)
            ]
            weight_sum = sum(self.deep_supervision_weights)
            self.deep_supervision_weights = [w / weight_sum for w in self.deep_supervision_weights]
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the nnU-Net model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor or tuple of output tensors (if deep supervision is used)
        """
        # Store skip connections
        skips = []
        
        # Encoder path
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)
        
        # Decoder path with skip connections
        decoder_outputs = [x]  # Start with bottleneck feature
        skip_idx = len(skips) - 1
        
        for decoder in self.decoders:
            skip_idx -= 1
            x = decoder(x, skips[skip_idx])
            decoder_outputs.insert(0, x)  # Add to beginning for correct order
        
        # Generate final output
        output = self.output(decoder_outputs[0])
        
        # Apply deep supervision if enabled
        if self.deep_supervision and self.training:
            deep_outputs = [output]
            
            # Get outputs from intermediate decoder layers
            for i, head in enumerate(self.deep_supervision_heads):
                # Get appropriate decoder output
                ds_input = decoder_outputs[i + 1]
                
                # Generate deep supervision output
                ds_output = head(ds_input)
                
                # Upsample to match the size of the main output
                target_size = output.shape[2:]
                ds_output = F.interpolate(ds_output, size=target_size, mode='trilinear', align_corners=False)
                
                deep_outputs.append(ds_output)
            
            return tuple(deep_outputs)
        
        return output