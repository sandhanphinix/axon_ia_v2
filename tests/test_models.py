"""
Tests for model implementations.

This module contains tests for the model architectures in axon_ia.
"""

import pytest
import torch

from axon_ia.models.model_factory import create_model


@pytest.mark.parametrize("architecture", [
    "unetr", 
    "swinunetr", 
    "nnunet", 
    "segresnet"
])
def test_model_creation(architecture):
    """Test creating models with different architectures."""
    # Create model
    model = create_model(
        architecture=architecture,
        in_channels=4,
        out_channels=1,
        img_size=(64, 64, 64),  # Smaller size for testing
        pretrained=False
    )
    
    # Check that model is created
    assert model is not None
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("architecture", [
    "unetr", 
    "swinunetr", 
    "nnunet", 
    "segresnet"
])
def test_model_forward_pass(architecture):
    """Test forward pass through different models."""
    # Create model
    model = create_model(
        architecture=architecture,
        in_channels=4,
        out_channels=1,
        img_size=(64, 64, 64),  # Smaller size for testing
        pretrained=False
    )
    
    # Create dummy input
    x = torch.randn(1, 4, 64, 64, 64)
    
    # Run forward pass
    with torch.no_grad():
        y = model(x)
    
    # Check output shape
    if isinstance(y, (tuple, list)):
        # For deep supervision, check main output
        assert y[0].shape[0] == 1
        assert y[0].shape[1] == 1
    else:
        assert y.shape[0] == 1
        assert y.shape[1] == 1


def test_model_ensemble():
    """Test model ensemble creation and forward pass."""
    # Create ensemble
    ensemble = create_model(
        architecture="ensemble",
        in_channels=4,
        out_channels=1,
        img_size=(64, 64, 64),
        models=[
            {
                "architecture": "unetr",
                "params": {
                    "in_channels": 4,
                    "out_channels": 1,
                    "img_size": (64, 64, 64),
                    "feature_size": 8  # Smaller for testing
                }
            },
            {
                "architecture": "segresnet",
                "params": {
                    "in_channels": 4,
                    "out_channels": 1,
                    "init_filters": 8  # Smaller for testing
                }
            }
        ],
        ensemble_method="mean"
    )
    
    # Create dummy input
    x = torch.randn(1, 4, 64, 64, 64)
    
    # Run forward pass
    with torch.no_grad():
        y = ensemble(x)
    
    # Check output shape
    assert y.shape == (1, 1, 64, 64, 64)