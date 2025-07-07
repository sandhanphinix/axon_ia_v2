"""
Configuration fixtures for Axon IA tests.

This module contains pytest fixtures used across multiple test files.
"""

import os
import sys
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.utils.nifti_utils import save_nifti


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def dummy_3d_data():
    """Create a dummy 3D volume and segmentation mask."""
    # Create random volume (128x128x64)
    volume = np.random.rand(128, 128, 64).astype(np.float32)
    
    # Create segmentation with a sphere
    x, y, z = np.ogrid[:128, :128, :64]
    mask = ((x - 64)**2 + (y - 64)**2 + ((z - 32) * 2)**2) <= 30**2
    mask = mask.astype(np.float32)
    
    return volume, mask


@pytest.fixture
def dummy_multi_modal_data():
    """Create a dummy multi-modal dataset."""
    # Create 4-channel volume (4x128x128x64)
    volume = np.random.rand(4, 128, 128, 64).astype(np.float32)
    
    # Create segmentation with multiple spheres
    x, y, z = np.ogrid[:128, :128, :64]
    mask1 = ((x - 50)**2 + (y - 50)**2 + ((z - 20) * 2)**2) <= 20**2
    mask2 = ((x - 80)**2 + (y - 80)**2 + ((z - 40) * 2)**2) <= 15**2
    mask = (mask1 | mask2).astype(np.float32)
    
    return volume, mask


@pytest.fixture
def dummy_nifti_dataset(temp_dir):
    """Create a dummy NIfTI dataset with the expected directory structure."""
    # Create train, val, test directories
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    test_dir = temp_dir / "test"
    
    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(exist_ok=True)
    
    # Create patient directories
    patient_ids = {
        "train": ["patient_001", "patient_002"],
        "val": ["patient_101"],
        "test": ["patient_201"]
    }
    
    for split, ids in patient_ids.items():
        for patient_id in ids:
            patient_dir = temp_dir / split / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            # Create dummy data
            volume = np.random.rand(128, 128, 64).astype(np.float32)
            
            # Create segmentation with a sphere
            x, y, z = np.ogrid[:128, :128, :64]
            mask = ((x - 64)**2 + (y - 64)**2 + ((z - 32) * 2)**2) <= 30**2
            mask = mask.astype(np.float32)
            
            # Save modalities
            for modality in ["flair", "t1", "t2", "dwi"]:
                save_nifti(volume, patient_dir / f"{modality}.nii.gz")
            
            # Save mask
            save_nifti(mask, patient_dir / "mask.nii.gz")
    
    return temp_dir


@pytest.fixture
def dummy_model():
    """Create a simple dummy model for testing."""
    # Define a simple 3D UNet-like model
    class DummyModel(torch.nn.Module):
        def __init__(self, in_channels=4, out_channels=1):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=2)
            )
            
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Conv3d(16, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(32),
                torch.nn.ReLU(),
            )
            
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
                torch.nn.Conv3d(16, 16, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(16),
                torch.nn.ReLU(),
            )
            
            self.output = torch.nn.Conv3d(16, out_channels, kernel_size=1)
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.bottleneck(x)
            x = self.decoder(x)
            x = self.output(x)
            return x
    
    return DummyModel()