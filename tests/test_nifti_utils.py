"""
Tests for NIfTI utilities.

This module contains tests for the NIfTI utility functions in axon_ia.
"""

import pytest
import numpy as np
from pathlib import Path

from axon_ia.utils.nifti_utils import (
    load_nifti,
    save_nifti,
    resample_nifti,
    get_brain_mask,
    combine_masks,
    compute_volume_stats,
    extract_roi
)


def test_save_and_load_nifti(temp_dir, dummy_3d_data):
    """Test saving and loading a NIfTI file."""
    volume, mask = dummy_3d_data
    
    # Define file path
    file_path = temp_dir / "test_volume.nii.gz"
    
    # Save the volume
    save_nifti(volume, file_path)
    
    # Check that the file was created
    assert file_path.exists()
    
    # Load the volume
    loaded_volume = load_nifti(file_path)
    
    # Check that the loaded volume has the same shape and values
    assert loaded_volume.shape == volume.shape
    assert np.allclose(loaded_volume, volume)
    
    # Test loading with metadata
    loaded_volume, meta = load_nifti(file_path, return_meta=True)
    assert 'affine' in meta
    assert 'header' in meta


def test_get_brain_mask(dummy_3d_data):
    """Test brain mask creation."""
    volume, _ = dummy_3d_data
    
    # Create brain mask
    mask = get_brain_mask(volume)
    
    # Check that the mask is binary
    assert np.unique(mask).tolist() == [0, 1]
    
    # Check that some voxels are labeled as brain
    assert np.sum(mask) > 0


def test_combine_masks(dummy_3d_data):
    """Test mask combination."""
    _, mask = dummy_3d_data
    
    # Create additional mask
    x, y, z = np.ogrid[:128, :128, :64]
    mask2 = ((x - 90)**2 + (y - 90)**2 + ((z - 40) * 2)**2) <= 20**2
    mask2 = mask2.astype(np.float32)
    
    # Combine masks using different methods
    union = combine_masks([mask, mask2], method='union')
    intersection = combine_masks([mask, mask2], method='intersection')
    majority = combine_masks([mask, mask2], method='majority')
    
    # Check results
    assert np.sum(union) >= np.sum(mask) and np.sum(union) >= np.sum(mask2)
    assert np.sum(intersection) <= np.sum(mask) and np.sum(intersection) <= np.sum(mask2)
    assert np.all((majority == 0) | (majority == 1))


def test_extract_roi(dummy_3d_data):
    """Test ROI extraction."""
    volume, mask = dummy_3d_data
    
    # Extract ROI
    roi, crop_indices = extract_roi(volume, mask)
    
    # Check that ROI is smaller than original volume
    assert roi.size < volume.size
    
    # Check that crop indices are valid
    for dim_slice in crop_indices:
        assert dim_slice[0] >= 0
        assert dim_slice[1] <= volume.shape[dim_slice[0]]