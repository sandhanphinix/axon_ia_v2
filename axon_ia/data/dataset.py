"""
Dataset implementations for medical image segmentation.

This module provides dataset classes for loading and processing
medical imaging data for segmentation tasks.
"""

import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Union, Callable
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from axon_ia.utils.logger import get_logger
from axon_ia.utils.nifti_utils import load_nifti

logger = get_logger()


class AxonDataset(Dataset):
    """
    Base dataset class for medical image segmentation.
    
    Features:
    - Multi-modal input support
    - Flexible directory structure
    - On-the-fly preprocessing and augmentation
    - Support for 2D and 3D data
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        modalities: List[str] = ["flair", "t1", "t2", "dwi"],
        target: str = "mask",
        transform: Optional[Callable] = None,
        preload: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.nii.gz",
        subsample: Optional[float] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing data
            split: Data split ('train', 'val', 'test')
            modalities: List of input modalities to use
            target: Name of target segmentation file
            transform: Transform to apply to samples
            preload: Whether to preload data into memory
            cache_dir: Directory to cache processed data
            pattern: Pattern to match files
            subsample: Fraction of data to use (0.0-1.0)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = modalities
        self.target = target
        self.transform = transform
        self.preload = preload
        self.pattern = pattern
        
        # Set up cache
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # Find samples
        self.samples = self._find_samples()
        
        # Subsample if requested
        if subsample is not None and 0.0 < subsample < 1.0:
            num_samples = max(1, int(len(self.samples) * subsample))
            random.seed(42)  # for reproducibility
            self.samples = random.sample(self.samples, num_samples)
            logger.info(f"Subsampled to {len(self.samples)} samples ({subsample:.2f})")
        
        # Preload data if requested
        self.data_cache = {}
        if self.preload:
            logger.info(f"Preloading data for {len(self.samples)} samples...")
            for i, sample_id in enumerate(self.samples):
                self.data_cache[sample_id] = self._load_sample(sample_id)
                if (i + 1) % 10 == 0:
                    logger.info(f"Preloaded {i+1}/{len(self.samples)} samples")
        
        logger.info(f"Created dataset with {len(self.samples)} samples in '{split}' split")
    
    def _find_samples(self) -> List[str]:
        """
        Find all valid samples in the dataset.
        
        A valid sample must have all required modalities and target.
        
        Returns:
            List of sample IDs
        """
        import glob
        
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Find all potential sample directories
        sample_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        # Filter to those that have all required files
        valid_samples = []
        for sample_dir in sample_dirs:
            # Check if all modalities and target exist
            has_all_files = True
            for modality in self.modalities:
                if not list(sample_dir.glob(f"{modality}{self.pattern[1:]}")):
                    has_all_files = False
                    break
            
            # Check target
            if has_all_files and not list(sample_dir.glob(f"{self.target}{self.pattern[1:]}")):
                has_all_files = False
            
            if has_all_files:
                valid_samples.append(sample_dir.name)
        
        if not valid_samples:
            logger.warning(f"No valid samples found in {split_dir}")
        
        return sorted(valid_samples)
    
    def _load_sample(self, sample_id: str) -> Dict[str, np.ndarray]:
        """
        Load a sample from disk.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Dictionary with image and mask arrays
        """
        sample_dir = self.root_dir / self.split / sample_id
        
        # Try to load from cache first
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self.split}_{sample_id}.npz"
            if cache_file.exists():
                try:
                    cached_data = np.load(cache_file, allow_pickle=True)
                    return {
                        "image": cached_data["image"],
                        "mask": cached_data["mask"],
                        "sample_id": sample_id
                    }
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")
        
        # Load modalities
        modality_arrays = []
        for modality in self.modalities:
            modality_files = list(sample_dir.glob(f"{modality}{self.pattern[1:]}"))
            if not modality_files:
                raise ValueError(f"Modality {modality} not found for sample {sample_id}")
            
            modality_data = load_nifti(modality_files[0])
            modality_arrays.append(modality_data)
        
        # Stack modalities to create multi-channel image
        image = np.stack(modality_arrays, axis=0)
        
        # Load target
        mask_files = list(sample_dir.glob(f"{self.target}{self.pattern[1:]}"))
        if not mask_files:
            raise ValueError(f"Target {self.target} not found for sample {sample_id}")
        
        mask = load_nifti(mask_files[0])
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)
        
        # Create sample
        sample = {
            "image": image.astype(np.float32),
            "mask": mask.astype(np.float32),
            "sample_id": sample_id
        }
        
        # Cache if needed
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self.split}_{sample_id}.npz"
            try:
                np.savez_compressed(
                    cache_file,
                    image=sample["image"],
                    mask=sample["mask"]
                )
            except Exception as e:
                logger.warning(f"Failed to cache sample: {e}")
        
        return sample
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image and mask tensors
        """
        sample_id = self.samples[idx]
        
        # Get sample data
        if sample_id in self.data_cache:
            sample = self.data_cache[sample_id].copy()
        else:
            sample = self._load_sample(sample_id)
        
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
        
        # Convert to tensors
        if isinstance(sample["image"], np.ndarray):
            sample["image"] = torch.from_numpy(sample["image"])
        if isinstance(sample["mask"], np.ndarray):
            sample["mask"] = torch.from_numpy(sample["mask"])
        
        return sample


class BrainTraumaDataset(AxonDataset):
    """
    Dataset for brain trauma segmentation.
    
    This specialized dataset is configured for brain trauma segmentation
    with appropriate modalities and preprocessing.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        modalities: List[str] = ["flair", "t1", "t2", "dwi"],
        target: str = "mask",
        transform: Optional[Callable] = None,
        preload: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.nii.gz",
        subsample: Optional[float] = None,
        include_metadata: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing data
            split: Data split ('train', 'val', 'test')
            modalities: List of input modalities to use
            target: Name of target segmentation file
            transform: Transform to apply to samples
            preload: Whether to preload data into memory
            cache_dir: Directory to cache processed data
            pattern: Pattern to match files
            subsample: Fraction of data to use (0.0-1.0)
            include_metadata: Whether to include patient metadata
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            modalities=modalities,
            target=target,
            transform=transform,
            preload=preload,
            cache_dir=cache_dir,
            pattern=pattern,
            subsample=subsample,
        )
        
        self.include_metadata = include_metadata
        
        # Load metadata if available and requested
        self.metadata = {}
        if include_metadata:
            metadata_file = self.root_dir / f"{split}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} samples")
    
    def _load_sample(self, sample_id: str) -> Dict[str, np.ndarray]:
        """
        Load a sample from disk with possible metadata.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Dictionary with image, mask arrays, and metadata
        """
        # Load basic sample
        sample = super()._load_sample(sample_id)
        
        # Add metadata if available
        if self.include_metadata and sample_id in self.metadata:
            sample["metadata"] = self.metadata[sample_id]
        
        return sample