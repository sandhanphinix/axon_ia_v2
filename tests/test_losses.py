"""
Tests for loss functions.

This module contains tests for the loss functions in axon_ia.
"""

import pytest
import torch

from axon_ia.losses.factory import create_loss_function
from axon_ia.losses.dice import DiceLoss, DiceCELoss
from axon_ia.losses.focal import FocalLoss
from axon_ia.losses.combo import ComboLoss


@pytest.mark.parametrize("loss_type", [
    "dice", 
    "dice_ce", 
    "focal", 
    "combo", 
    "boundary"
])
def test_loss_creation(loss_type):
    """Test creating loss functions with different types."""
    # Create loss function
    loss_fn = create_loss_function(loss_type)
    
    # Check that loss function is created
    assert loss_fn is not None
    assert isinstance(loss_fn, torch.nn.Module)


@pytest.mark.parametrize("loss_class, loss_kwargs", [
    (DiceLoss, {}),
    (DiceCELoss, {}),
    (FocalLoss, {"gamma": 2.0}),
    (ComboLoss, {"dice_weight": 1.0, "focal_weight": 1.0})
])
def test_loss_backward(loss_class, loss_kwargs):
    """Test that loss functions can compute gradients."""
    # Create model with trainable parameters
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 1, kernel_size=3, padding=1)
    )
    
    # Create loss function
    loss_fn = loss_class(**loss_kwargs)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create dummy input and target
    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
    
    # Forward pass
    pred = model(x)
    loss = loss_fn(pred, target)
    
    # Check that loss is valid
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients are computed
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_dice_loss():
    """Test Dice loss calculation."""
    # Create loss function
    loss_fn = DiceLoss()
    
    # Create predictions and targets
    pred = torch.zeros(2, 1, 16, 16, 16)
    target = torch.zeros(2, 1, 16, 16, 16)
    
    # Perfect match
    pred[0, 0, 4:12, 4:12, 4:12] = 1.0
    target[0, 0, 4:12, 4:12, 4:12] = 1.0
    
    # Complete mismatch
    pred[1, 0, 4:12, 4:12, 4:12] = 1.0
    target[1, 0, 12:16, 12:16, 12:16] = 1.0
    
    # Apply sigmoid
    pred = torch.sigmoid(20 * (pred - 0.5))
    
    # Calculate loss
    loss = loss_fn(pred, target)
    
    # For perfect match, Dice score is 1 so loss is 0
    # For complete mismatch, Dice score is 0 so loss is 1
    # Average loss should be around 0.5
    assert 0.4 < loss.item() < 0.6