"""
Model factory for creating neural network models.

This module provides a factory function for creating
different neural network architectures for segmentation.
"""

import torch
import torch.nn as nn
from typing import Union

from axon_ia.models.unetr import UNETR
from axon_ia.models.nnunet import NNUNet
from axon_ia.models.swinunetr import SwinUNETR
from axon_ia.models.segresnet import SegResNet
from axon_ia.models.ensemble import ModelEnsemble, ResidualUNet3D, DenseUNet3D, MultiScaleUNet
from axon_ia.utils.logger import get_logger

logger = get_logger()


def create_model(
    architecture: str,
    in_channels: int = 4,
    out_channels: int = 1,
    img_size: Union[int, tuple] = (128, 128, 128),
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a neural network model based on the specified architecture.
    
    Args:
        architecture: Model architecture name
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Input image size
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific parameters
        
    Returns:
        Neural network model
    """
    architecture = architecture.lower()
    
    # Convert img_size to tuple if it's an integer
    if isinstance(img_size, int):
        img_size = (img_size, img_size, img_size)
    
    # Handle patch_size conversion for UNETR
    patch_size = kwargs.get("patch_size", (16, 16, 16))
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    
    # UNETR model
    if architecture == "unetr":
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=kwargs.get("feature_size", 16),
            hidden_size=kwargs.get("hidden_size", 768),
            mlp_dim=kwargs.get("mlp_dim", 3072),
            num_heads=kwargs.get("num_heads", 12),
            pos_embed=kwargs.get("pos_embed", "sincos"),  # Add pos_embed parameter
            norm_name=kwargs.get("norm_name", "instance"),
            conv_block=kwargs.get("conv_block", True),
            res_block=kwargs.get("res_block", True),
            dropout_rate=kwargs.get("dropout_rate", 0.0),
            use_deep_supervision=kwargs.get("use_deep_supervision", False),
            patch_size=patch_size  # Use the processed patch_size
        )
    
    # SwinUNETR model
    elif architecture == "swinunetr" or architecture == "swin_unetr":
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kwargs.get("feature_size", 48),
            drop_rate=kwargs.get("drop_rate", 0.0),
            attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
            dropout_path_rate=kwargs.get("dropout_path_rate", 0.0),
            use_checkpoint=kwargs.get("use_checkpoint", False),
            use_deep_supervision=kwargs.get("use_deep_supervision", False)
        )
    
    # nnUNet model
    elif architecture == "nnunet":
        model = NNUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_channels=kwargs.get("feature_channels", [32, 64, 128, 256, 320, 320]),
            strides=kwargs.get("strides", [1, 2, 2, 2, 2, 2]),
            kernel_size=kwargs.get("kernel_size", 3),
            norm_name=kwargs.get("norm_name", "instance"),
            deep_supervision=kwargs.get("deep_supervision", False),
            deep_supr_num=kwargs.get("deep_supr_num", 2),
            res_block=kwargs.get("res_block", True),
            attention_gate=kwargs.get("attention_gate", False)
        )
    
    # SegResNet model
    elif architecture == "segresnet":
        # Map parameters to match our SegResNet implementation
        model = SegResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=kwargs.get("init_filters", 16),
            blocks_down=kwargs.get("blocks_down", [1, 2, 2, 4]),
            blocks_up=kwargs.get("blocks_up", [1, 1, 1]),
            norm_name=kwargs.get("norm_name", kwargs.get("norm", "instance")),  # Support both names
            upsample_mode=kwargs.get("upsample_mode", "transpose"),
            deep_supervision=kwargs.get("deep_supervision", False)            # Note: spatial_dims, dropout_prob, use_conv_final, use_attention are ignored for compatibility
        )
    
    # ResidualUNet3D model (from ensemble.py)
    elif architecture == "residual_unet":
        model = ResidualUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=kwargs.get("features", [32, 64, 128, 256, 512]),
            dropout=kwargs.get("dropout", 0.1),
            attention_gates=kwargs.get("attention_gates", True),
            deep_supervision=kwargs.get("deep_supervision", True),
            residual_connections=kwargs.get("residual_connections", True)
            # Note: squeeze_excitation is ignored for compatibility
        )
    
    # MultiScaleUNet model (from ensemble.py)
    elif architecture == "multiscale_unet":
        model = MultiScaleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=kwargs.get("base_features", 32),
            growth_rate=kwargs.get("growth_rate", 16),
            num_layers=kwargs.get("num_layers", [4, 5, 7, 10, 12]),
            dropout=kwargs.get("dropout", 0.1),
            multiscale_features=kwargs.get("multiscale_features", True),
            dense_connections=kwargs.get("dense_connections", True),
            deep_supervision=kwargs.get("deep_supervision", True),
            pyramid_pooling=kwargs.get("pyramid_pooling", True)
        )

    # Model ensemble
    elif architecture == "ensemble":
        # Create ensemble from individual models
        model_configs = kwargs.get("models", [])
        models = []
        
        for config in model_configs:
            model_arch = config.get("architecture", "unetr")
            model_kwargs = config.get("params", {})
            model_checkpoint = config.get("checkpoint", None)
            
            # Create model
            individual_model = create_model(
                model_arch,
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=img_size,
                **model_kwargs
            )
            
            # Load checkpoint if provided
            if model_checkpoint:
                try:
                    checkpoint = torch.load(model_checkpoint, map_location="cpu")
                    if "model_state_dict" in checkpoint:
                        individual_model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        individual_model.load_state_dict(checkpoint)
                    logger.info(f"Loaded checkpoint for {model_arch} from {model_checkpoint}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint for {model_arch}: {e}")
            
            models.append(individual_model)
        
        # Create ensemble
        model = ModelEnsemble(
            models=models,
            ensemble_method=kwargs.get("ensemble_method", "mean"),
            weights=kwargs.get("weights", None)
        )
    
    else:
        available = ["unetr", "swinunetr", "nnunet", "segresnet", "residual_unet", "multiscale_unet", "ensemble"]
        raise ValueError(f"Unknown architecture: {architecture}. Available: {available}")
    
    # # Load pretrained weights if requested
    # if pretrained and architecture != "ensemble":
    #     try:
    #         from axon_ia.utils.model_weights import load_pretrained_weights
    #         load_pretrained_weights(model, architecture)
    #         logger.info(f"Loaded pretrained weights for {architecture}")
    #     except Exception as e:
    #         logger.warning(f"Failed to load pretrained weights: {e}") 
    # Log model creation
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {architecture} model with {num_params:,} parameters")
    
    return model