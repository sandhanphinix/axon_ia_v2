"""
Ensemble model implementations for medical image segmentation.

This module provides various segmentation models and ensemble methods
optimized for medical imaging tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

try:
    from monai.networks.nets import SwinUNETR, UNETR, SegResNet, AttentionUnet
    from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Some models may not work.")


class ResidualUNet3D(nn.Module):
    """
    3D U-Net with residual connections and attention gates.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        dropout: float = 0.1,
        attention_gates: bool = True,
        deep_supervision: bool = True,
        residual_connections: bool = True
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.residual_connections = residual_connections
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoder_layers.append(
                self._make_layer(in_ch, feature, dropout, residual_connections)
            )
            if i < len(features) - 1:
                self.pool_layers.append(nn.MaxPool3d(2))
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention_gates else None
        
        for i in range(len(features) - 2, -1, -1):
            self.upconv_layers.append(
                nn.ConvTranspose3d(features[i+1], features[i], 2, 2)
            )
            
            if attention_gates:
                self.attention_gates.append(
                    AttentionGate3D(features[i], features[i], features[i]//2)  # Fixed: Both g and x have same channels after upconv
                )
            
            self.decoder_layers.append(
                self._make_layer(features[i]*2, features[i], dropout, residual_connections)
            )
        
        # Output layers
        self.final_conv = nn.Conv3d(features[0], out_channels, 1)
        
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv3d(features[i], out_channels, 1) 
                for i in range(len(features)-1)
            ])
    
    def _make_layer(self, in_channels, out_channels, dropout, residual):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout3d(dropout))
        layers.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        
        if residual and in_channels == out_channels:
            return ResidualBlock3D(nn.Sequential(*layers))
        else:
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        for i, (layer, pool) in enumerate(zip(self.encoder_layers[:-1], self.pool_layers)):
            x = layer(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.encoder_layers[-1](x)
        
        # Decoder
        deep_outputs = []
        for i, (upconv, decoder) in enumerate(zip(self.upconv_layers, self.decoder_layers)):
            x = upconv(x)
            
            # Attention gate
            skip = encoder_features[-(i+1)]
            if self.attention_gates:
                skip = self.attention_gates[i](skip, x)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
            
            if self.deep_supervision and i < len(self.deep_outputs):
                deep_outputs.append(self.deep_outputs[i](x))
        
        # Final output
        output = self.final_conv(x)
        
        if self.deep_supervision and self.training:
            return [output] + deep_outputs
        else:
            return output


class ResidualBlock3D(nn.Module):
    """3D Residual block."""
    
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        return F.relu(x + self.layers(x))


class AttentionGate3D(nn.Module):
    """3D Attention gate mechanism."""
    
    def __init__(self, g_channels, x_channels, int_channels):
        super().__init__()
        self.W_g = nn.Conv3d(g_channels, int_channels, 1)
        self.W_x = nn.Conv3d(x_channels, int_channels, 1)
        self.psi = nn.Conv3d(int_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple segmentation models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting_strategy: str = "soft",
        tta_enabled: bool = True
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.voting_strategy = voting_strategy
        self.tta_enabled = tta_enabled
        
        assert len(self.weights) == len(self.models)
        assert abs(sum(self.weights) - 1.0) < 1e-6
    
    def forward(self, x):
        if self.training:
            # During training, only use one model at a time for efficiency
            model_idx = torch.randint(0, len(self.models), (1,)).item()
            return self.models[model_idx](x)
        else:
            # During inference, use ensemble
            outputs = []
            
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    if self.tta_enabled:
                        output = self._test_time_augmentation(model, x)
                    else:
                        output = model(x)
                    outputs.append(output)
            
            return self._combine_outputs(outputs)
    
    def _test_time_augmentation(self, model, x):
        """Apply test-time augmentation."""
        predictions = []
        
        # Original
        predictions.append(model(x))
        
        # Horizontal flip
        x_flip = torch.flip(x, dims=[2])
        pred_flip = model(x_flip)
        predictions.append(torch.flip(pred_flip, dims=[2]))
        
        # Vertical flip
        x_flip = torch.flip(x, dims=[3])
        pred_flip = model(x_flip)
        predictions.append(torch.flip(pred_flip, dims=[3]))
        
        # Depth flip
        x_flip = torch.flip(x, dims=[4])
        pred_flip = model(x_flip)
        predictions.append(torch.flip(pred_flip, dims=[4]))
        
        return torch.mean(torch.stack(predictions), dim=0)
    
    def _combine_outputs(self, outputs):
        """Combine outputs from multiple models."""
        if self.voting_strategy == "soft":
            # Weighted average
            weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
            return torch.sum(torch.stack(weighted_outputs), dim=0)
        
        elif self.voting_strategy == "hard":
            # Hard voting (majority vote)
            binary_outputs = [torch.sigmoid(out) > 0.5 for out in outputs]
            return torch.sum(torch.stack([bo.float() for bo in binary_outputs]), dim=0) > len(outputs) / 2
        
        elif self.voting_strategy == "weighted":
            # Confidence-weighted voting
            confidences = [torch.max(torch.sigmoid(out), 1 - torch.sigmoid(out)) for out in outputs]
            normalized_weights = [conf / torch.sum(torch.stack(confidences), dim=0) for conf in confidences]
            
            weighted_outputs = [w * out for w, out in zip(normalized_weights, outputs)]
            return torch.sum(torch.stack(weighted_outputs), dim=0)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")


class ModelEnsemble(nn.Module):
    """
    Model ensemble class compatible with model_factory interface.
    This is a wrapper around EnsembleModel to match the expected API.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "mean",
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        # Map ensemble_method to voting_strategy
        voting_strategy_map = {
            "mean": "soft",
            "soft": "soft", 
            "hard": "hard",
            "weighted": "weighted"
        }
        
        voting_strategy = voting_strategy_map.get(ensemble_method, "soft")
        
        # Use the existing EnsembleModel implementation
        self.ensemble = EnsembleModel(
            models=models,
            weights=weights,
            voting_strategy=voting_strategy,
            tta_enabled=False  # Disable TTA by default for training
        )
    
    def forward(self, x):
        return self.ensemble(x)
    
    def train(self, mode: bool = True):
        """Override train method to ensure proper training mode."""
        super().train(mode)
        self.ensemble.train(mode)
        return self
    
    def eval(self):
        """Override eval method to ensure proper evaluation mode."""
        super().eval()
        self.ensemble.eval()
        return self


class DenseUNet3D(nn.Module):
    """
    3D DenseNet-based U-Net with multi-scale supervision and pyramid pooling.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: List[int] = [4, 6, 12, 8],
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.1,
        multiscale_supervision: bool = True,
        pyramid_pooling: bool = True,
        aspp_dilations: List[int] = [6, 12, 18]
    ):
        super().__init__()
        self.multiscale_supervision = multiscale_supervision
        self.pyramid_pooling = pyramid_pooling
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer3D(num_features, num_features // 2)
                self.transition_layers.append(trans)
                num_features = num_features // 2
        
        # ASPP module
        if pyramid_pooling:
            self.aspp = ASPP3D(num_features, 256, aspp_dilations)
            aspp_features = 256
        else:
            aspp_features = num_features
        
        # Final output layer
        self.final_conv = nn.Conv3d(aspp_features, out_channels, 1)
    
    def forward(self, x):
        features = self.features(x)
        
        for i, (block, trans) in enumerate(zip(self.dense_blocks[:-1], self.transition_layers)):
            features = block(features)
            features = trans(features)
        
        features = self.dense_blocks[-1](features)
        
        if self.pyramid_pooling:
            features = self.aspp(features)
        
        return self.final_conv(features)


class SqueezeExcitation3D(nn.Module):
    """3D Squeeze and Excitation block."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class DenseBlock3D(nn.Module):
    """3D Dense block for DenseNet."""
    
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer3D(nn.Module):
    """3D Dense layer."""
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, 1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        
        self.drop_rate = drop_rate
    
    def forward(self, x):
        bottleneck_output = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return new_features


class TransitionLayer3D(nn.Module):
    """3D Transition layer for DenseNet."""
    
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features, 1, bias=False)
        self.pool = nn.AvgPool3d(2, stride=2)
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling."""
    
    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolutions
        for dilation in dilations:
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),  # Use InstanceNorm instead of BatchNorm for single-sample cases
            nn.ReLU(inplace=True)
        )
        
        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv3d(out_channels * (len(dilations) + 2), out_channels, 1),
            nn.InstanceNorm3d(out_channels),  # Use InstanceNorm for stability
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x):
        features = []
        
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='trilinear', align_corners=False)
        features.append(global_feat)
        
        return self.final_conv(torch.cat(features, dim=1))


class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net with pyramid pooling and dense connections.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 32,
        growth_rate: int = 16,
        num_layers: List[int] = [4, 5, 7, 10, 12],
        dropout: float = 0.1,
        multiscale_features: bool = True,
        dense_connections: bool = True,
        deep_supervision: bool = True,
        pyramid_pooling: bool = True
    ):
        super().__init__()
        self.multiscale_features = multiscale_features
        self.dense_connections = dense_connections
        self.deep_supervision = deep_supervision
        self.pyramid_pooling = pyramid_pooling
        
        # Multi-scale input processing
        if multiscale_features:
            self.multiscale_conv = nn.ModuleList([
                nn.Conv3d(in_channels, base_features//4, 1),  # 1x1
                nn.Conv3d(in_channels, base_features//4, 3, padding=1),  # 3x3
                nn.Conv3d(in_channels, base_features//4, 5, padding=2),  # 5x5
                nn.Conv3d(in_channels, base_features//4, 7, padding=3),  # 7x7
            ])
            encoder_in_channels = base_features
        else:
            self.initial_conv = nn.Conv3d(in_channels, base_features, 3, padding=1)
            encoder_in_channels = base_features
        
        # Encoder with dense connections
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        current_channels = encoder_in_channels
        for i, layers in enumerate(num_layers):
            if dense_connections:
                block = DenseBlock3D(layers, current_channels, 4, growth_rate, dropout)
                current_channels += layers * growth_rate
            else:
                block = self._make_conv_block(current_channels, base_features * (2**i), dropout)
                current_channels = base_features * (2**i)
            
            self.encoder_blocks.append(block)
            
            if i < len(num_layers) - 1:
                self.pool_layers.append(nn.MaxPool3d(2))
        
        # Pyramid pooling
        if pyramid_pooling:
            self.ppm = ASPP3D(current_channels, current_channels//4, [1, 2, 3, 6])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(len(num_layers)-2, -1, -1):
            up_channels = current_channels
            skip_channels = base_features * (2**i) if not dense_connections else encoder_in_channels + sum(num_layers[j] * growth_rate for j in range(i+1))
            
            self.upconv_layers.append(
                nn.ConvTranspose3d(up_channels, up_channels//2, 2, 2)
            )
            
            if dense_connections:
                block = DenseBlock3D(num_layers[i], up_channels//2 + skip_channels, 4, growth_rate, dropout)
                current_channels = up_channels//2 + skip_channels + num_layers[i] * growth_rate
            else:
                block = self._make_conv_block(up_channels//2 + skip_channels, up_channels//2, dropout)
                current_channels = up_channels//2
            
            self.decoder_blocks.append(block)
        
        # Output
        self.final_conv = nn.Conv3d(current_channels, out_channels, 1)
        
        # Deep supervision
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv3d(base_features * (2**i), out_channels, 1)
                for i in range(len(num_layers)-1)
            ])
    
    def _make_conv_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Multi-scale input processing
        if self.multiscale_features:
            multiscale_features = [conv(x) for conv in self.multiscale_conv]
            x = torch.cat(multiscale_features, dim=1)
        else:
            x = self.initial_conv(x)
        
        # Encoder
        encoder_features = []
        for i, (block, pool) in enumerate(zip(self.encoder_blocks[:-1], self.pool_layers)):
            x = block(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.encoder_blocks[-1](x)
        
        # Pyramid pooling
        if self.pyramid_pooling:
            x = self.ppm(x)
        
        # Decoder
        deep_outputs = []
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            x = upconv(x)
            
            skip = encoder_features[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
            
            if self.deep_supervision and i < len(self.deep_outputs):
                deep_outputs.append(self.deep_outputs[i](x))
        
        # Final output
        output = self.final_conv(x)
        
        if self.deep_supervision and self.training:
            return [output] + deep_outputs
        else:
            return output


def create_model(architecture: str, **params) -> nn.Module:
    """
    Factory function to create different model architectures.
    """
    if not MONAI_AVAILABLE and architecture in ["swinunetr", "unetr", "segresnet"]:
        raise ImportError(f"MONAI is required for {architecture}. Please install: pip install monai")
    
    if architecture == "swinunetr":
        return SwinUNETR(**params)
    elif architecture == "unetr":
        return UNETR(**params)
    elif architecture == "segresnet":
        return SegResNet(**params)
    elif architecture == "residual_unet":
        return ResidualUNet3D(**params)
    elif architecture == "dense_unet_3d":
        return DenseUNet3D(**params)
    elif architecture == "multiscale_unet":
        return MultiScaleUNet(**params)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Available: swinunetr, unetr, segresnet, residual_unet, dense_unet_3d, multiscale_unet")


def create_ensemble(model_configs: List[Dict], ensemble_config: Dict):
    """
    Create an ensemble of models from configuration.
    """
    models = []
    
    for config in model_configs:
        model = create_model(
            architecture=config["architecture"],
            **config.get("params", {})
        )
        models.append(model)
    
    ensemble = EnsembleModel(
        models=models,
        weights=ensemble_config.get("weights"),
        voting_strategy=ensemble_config.get("voting_strategy", "soft"),
        tta_enabled=ensemble_config.get("tta_enabled", True)
    )
    
    return ensemble
