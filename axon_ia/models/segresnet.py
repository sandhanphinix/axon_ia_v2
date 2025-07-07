"""
SegResNet model implementation.

This module provides an implementation of the SegResNet architecture,
which is a 3D ResNet-based segmentation model.

Reference:
    Myronenko, A. "3D MRI brain tumor segmentation using autoencoder regularization"
    https://arxiv.org/abs/1810.11654
"""

from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block for SegResNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_name: str = "instance",
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernels
            norm_name: Type of normalization layer
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
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.norm1 = norm_layer(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.norm2 = norm_layer(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                norm_layer(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Main path
        res = self.residual(x)
        
        # Residual path
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        
        # Add residual connection
        x += res
        x = self.relu(x)
        
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder for SegResNet.
    """
    
    def __init__(
        self,
        in_channels: int,
        init_filters: int,
        blocks_down: List[int],
        norm_name: str = "instance",
    ):
        """
        Initialize ResNet encoder.
        
        Args:
            in_channels: Number of input channels
            init_filters: Number of initial filters
            blocks_down: Number of residual blocks in each encoder stage
            norm_name: Type of normalization layer
        """
        super().__init__()
        
        # Initial convolution
        self.conv_init = nn.Conv3d(
            in_channels, init_filters, kernel_size=3, padding=1
        )
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_filters = init_filters
        out_filters = init_filters
        
        for i, num_blocks in enumerate(blocks_down):
            # Increase filters by factor of 2 after first stage
            if i > 0:
                out_filters *= 2
            
            # Add downsampling convolution
            self.downsamplers.append(
                nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            
            # Add blocks for this stage
            stage_blocks = []
            stage_blocks.append(ResBlock(out_filters, out_filters, norm_name=norm_name))
            
            for _ in range(num_blocks - 1):
                stage_blocks.append(ResBlock(out_filters, out_filters, norm_name=norm_name))
            
            self.encoders.append(nn.Sequential(*stage_blocks))
            in_filters = out_filters
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            List of features from each encoder stage
        """
        # Initial convolution
        x = self.conv_init(x)
        
        # Store features for skip connections
        features = [x]
        
        # Encoder path
        for down, encoder in zip(self.downsamplers, self.encoders):
            # Downsample
            x = down(x)
            
            # Process through blocks
            x = encoder(x)
            
            # Store features
            features.append(x)
        
        return features


class UpsampleBlock(nn.Module):
    """
    Upsampling block for SegResNet decoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_name: str = "instance",
        upsample_mode: str = "transpose",
    ):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            norm_name: Type of normalization layer
            upsample_mode: Upsampling mode ('transpose', 'trilinear')
        """
        super().__init__()
        
        if upsample_mode == "transpose":
            # Transposed convolution
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            # Trilinear interpolation followed by 1x1 convolution
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x: Input tensor
            
        Returns:
            Upsampled tensor
        """
        return self.upsample(x)


class SegResNet(nn.Module):
    """
    SegResNet model for medical image segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        init_filters: int = 16,
        blocks_down: List[int] = [1, 2, 2, 4],
        blocks_up: List[int] = [1, 1, 1],
        norm_name: str = "instance",
        upsample_mode: str = "transpose",
        deep_supervision: bool = False,
    ):
        """
        Initialize SegResNet model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (classes)
            init_filters: Number of initial filters
            blocks_down: Number of residual blocks in each encoder stage
            blocks_up: Number of residual blocks in each decoder stage
            norm_name: Type of normalization layer
            upsample_mode: Upsampling mode ('transpose', 'trilinear')
            deep_supervision: Whether to use deep supervision
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Create encoder
        self.encoder = ResNetEncoder(
            in_channels, init_filters, blocks_down, norm_name
        )
        
        # Calculate number of filters at each level
        encoder_filters = [init_filters]
        for i in range(len(blocks_down)):
            if i == 0:
                encoder_filters.append(init_filters)
            else:
                encoder_filters.append(encoder_filters[-1] * 2)
        
        # Create decoder
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        # Iterate through decoder stages in reverse order
        for i in range(len(blocks_up)):
            # Calculate current level in the network
            level = len(blocks_down) - i
            
            # Create upsampler
            self.upsamplers.append(
                UpsampleBlock(
                    encoder_filters[level],
                    encoder_filters[level - 1],
                    norm_name,
                    upsample_mode
                )
            )
            
            # Create decoder blocks
            decoder_blocks = []
            decoder_blocks.append(
                ResBlock(
                    encoder_filters[level - 1] * 2,  # Skip connection doubles channels
                    encoder_filters[level - 1],
                    norm_name=norm_name
                )
            )
            
            for _ in range(blocks_up[i] - 1):
                decoder_blocks.append(
                    ResBlock(
                        encoder_filters[level - 1],
                        encoder_filters[level - 1],
                        norm_name=norm_name
                    )
                )
            
            self.decoders.append(nn.Sequential(*decoder_blocks))
        
        # Output convolution
        self.output = nn.Conv3d(encoder_filters[0], out_channels, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(len(blocks_up) - 1):
                level = len(blocks_down) - i - 1
                self.deep_supervision_heads.append(
                    nn.Conv3d(encoder_filters[level], out_channels, kernel_size=1)
                )
            
            # Set deep supervision weights (can be tuned)
            self.deep_supervision_weights = [
                1.0 / (2**i) for i in range(len(blocks_up))
            ]
            weight_sum = sum(self.deep_supervision_weights)
            self.deep_supervision_weights = [w / weight_sum for w in self.deep_supervision_weights]
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the SegResNet model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor or tuple of output tensors (if deep supervision is used)
        """
        # Encoder path
        features = self.encoder(x)
        
        # Bottleneck is the last feature
        x = features[-1]
        
        # Decoder path with skip connections
        decoder_features = []
        
        for i, (up, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            # Current level in the network
            level = len(features) - i - 1
            
            # Upsample
            x = up(x)
            
            # Concatenate with skip connection
            x = torch.cat([x, features[level - 1]], dim=1)
            
            # Process through decoder blocks
            x = decoder(x)
            
            # Store decoder features for deep supervision
            decoder_features.append(x)
        
        # Generate final output
        output = self.output(x)
        
        # Apply deep supervision if enabled
        if self.deep_supervision and self.training:
            deep_outputs = [output]
            
            # Process decoder features
            for i, head in enumerate(self.deep_supervision_heads):
                # Skip the first decoder output which is already processed
                feature = decoder_features[-i-2]
                
                # Apply supervision head
                ds_output = head(feature)
                
                # Upsample to match the size of the main output
                target_size = output.shape[2:]
                ds_output = F.interpolate(ds_output, size=target_size, mode='trilinear', align_corners=False)
                
                deep_outputs.append(ds_output)
            
            return tuple(deep_outputs)
        
        return output