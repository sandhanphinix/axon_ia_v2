"""
Tests for dataset implementations.

This module contains tests for the dataset classes in axon_ia.
"""

import pytest
import torch
from pathlib import Path

from axon_ia.data.dataset import AxonDataset, BrainTraumaDataset


def test_axon_dataset_creation(dummy_nifti_dataset):
    """Test creating an AxonDataset."""
    # Create dataset
    dataset = AxonDataset(
        root_dir=dummy_nifti_dataset,
        split="train",
        modalities=["flair", "t1", "t2", "dwi"],
        target="mask"
    )
    
    # Check that dataset is created
    assert dataset is not None
    assert len(dataset) == 2  # Two patients in train split


def test_axon_dataset_getitem(dummy_nifti_dataset):
    """Test getting an item from AxonDataset."""
    # Create dataset
    dataset = AxonDataset(
        root_dir=dummy_nifti_dataset,
        split="train",
        modalities=["flair", "t1", "t2", "dwi"],
        target="mask"
    )
    
    # Get an item
    sample = dataset[0]
    
    # Check that sample has expected keys and shapes
    assert "image" in sample
    assert "mask" in sample
    assert "sample_id" in sample
    
    # Check that image has 4 channels (4 modalities)
    assert sample["image"].shape[0] == 4
    
    # Check that mask is a tensor
    assert isinstance(sample["mask"], torch.Tensor)


def test_brain_trauma_dataset(dummy_nifti_dataset):
    """Test creating a BrainTraumaDataset."""
    # Create dataset
    dataset = BrainTraumaDataset(
        root_dir=dummy_nifti_dataset,
        split="train",
        modalities=["flair", "t1", "t2", "dwi"],
        target="mask",
        include_metadata=True
    )
    
    # Check that dataset is created
    assert dataset is not None
    assert len(dataset) == 2  # Two patients in train split
    
    # Get an item
    sample = dataset[0]
    
    # Check that sample has expected keys
    assert "image" in sample
    assert "mask" in sample
    assert "sample_id" in sample