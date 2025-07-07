"""
Custom samplers for balanced batch construction.

This module provides custom samplers for creating balanced batches,
particularly useful for dealing with class imbalance in medical segmentation.
"""

import random
from typing import Dict, List, Optional, Tuple, Union, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from axon_ia.utils.logger import get_logger

logger = get_logger()


class BalancedSampler(Sampler):
    """
    Sampler that balances positive and negative examples.
    
    This sampler ensures that each batch contains a balanced
    number of positive (with lesions) and negative (without lesions)
    examples, which is crucial for training with highly imbalanced datasets.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        pos_fraction: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            pos_fraction: Fraction of positive examples in each batch
            shuffle: Whether to shuffle within positive and negative groups
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Determine which samples have positive masks
        self.positive_indices = []
        self.negative_indices = []
        
        logger.info("Analyzing dataset for balanced sampling...")
        for i, sample_id in enumerate(dataset.samples):
            # Check if sample has been preloaded
            if hasattr(dataset, 'data_cache') and sample_id in dataset.data_cache:
                mask = dataset.data_cache[sample_id]["mask"]
            else:
                # Load mask to check
                try:
                    mask = dataset._load_sample(sample_id)["mask"]
                except Exception as e:
                    logger.warning(f"Could not check if sample {sample_id} is positive: {e}")
                    continue
            
            # Check if mask has any positive voxels
            has_positive = np.any(mask > 0) if isinstance(mask, np.ndarray) else torch.any(mask > 0)
            
            if has_positive:
                self.positive_indices.append(i)
            else:
                self.negative_indices.append(i)
        
        logger.info(f"Found {len(self.positive_indices)} positive and {len(self.negative_indices)} negative samples")
        
        # Calculate number of samples per epoch
        pos_per_batch = max(1, int(batch_size * pos_fraction))
        neg_per_batch = batch_size - pos_per_batch
        
        # Check if we have enough samples of each type
        if len(self.positive_indices) < pos_per_batch:
            logger.warning(
                f"Not enough positive samples ({len(self.positive_indices)}) "
                f"for desired pos_per_batch ({pos_per_batch}). "
                f"Will use all available positive samples."
            )
        if len(self.negative_indices) < neg_per_batch:
            logger.warning(
                f"Not enough negative samples ({len(self.negative_indices)}) "
                f"for desired neg_per_batch ({neg_per_batch}). "
                f"Will use all available negative samples."
            )
        
        # Calculate total number of batches
        self.num_batches = min(
            len(self.positive_indices) // pos_per_batch if pos_per_batch > 0 else float('inf'),
            len(self.negative_indices) // neg_per_batch if neg_per_batch > 0 else float('inf')
        )
        
        if self.num_batches == 0:
            logger.error("Cannot create any complete batches with current settings!")
            raise ValueError("Cannot create any complete batches with current settings")
        
        # Adjust total length based on drop_last
        self.total_size = self.num_batches * batch_size
        
        # Set batch composition
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = neg_per_batch
        
        logger.info(f"Created balanced sampler with {self.num_batches} batches per epoch")
        logger.info(f"Each batch will have {pos_per_batch} positive and {neg_per_batch} negative samples")
    
    def __iter__(self) -> Iterator[int]:
        """
        Create iterator over indices.
        
        Returns:
            Iterator over dataset indices
        """
        # Shuffle indices if requested
        if self.shuffle:
            pos_indices = self.positive_indices.copy()
            neg_indices = self.negative_indices.copy()
            random.shuffle(pos_indices)
            random.shuffle(neg_indices)
        else:
            pos_indices = self.positive_indices
            neg_indices = self.negative_indices
        
        # Create batches
        for i in range(self.num_batches):
            # Get indices for this batch
            batch_pos = pos_indices[i*self.pos_per_batch:(i+1)*self.pos_per_batch]
            batch_neg = neg_indices[i*self.neg_per_batch:(i+1)*self.neg_per_batch]
            
            # Combine and yield batch indices
            batch_indices = batch_pos + batch_neg
            random.shuffle(batch_indices)  # Shuffle within batch
            
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        """
        Get sampler length.
        
        Returns:
            Total number of samples per epoch
        """
        return self.total_size


class WeightedSampler(Sampler):
    """
    Weighted sampler for emphasizing difficult examples.
    
    This sampler assigns weights to samples based on their difficulty,
    allowing the model to focus more on challenging cases.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        weights: Union[List[float], np.ndarray, torch.Tensor],
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Initialize weighted sampler.
        
        Args:
            dataset: Dataset to sample from
            weights: Sample weights (higher means more likely to be sampled)
            num_samples: Number of samples per epoch (default: len(dataset))
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
    
    def __iter__(self) -> Iterator[int]:
        """
        Create iterator over indices.
        
        Returns:
            Iterator over dataset indices
        """
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement
        ).tolist()
        return iter(indices)
    
    def __len__(self) -> int:
        """
        Get sampler length.
        
        Returns:
            Number of samples per epoch
        """
        return self.num_samples


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler for balanced classes.
    
    This sampler creates batches with samples from all represented classes,
    ensuring that each batch contains a diverse set of examples.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        class_indices: Dict[int, List[int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize stratified batch sampler.
        
        Args:
            dataset: Dataset to sample from
            class_indices: Dictionary mapping class index to list of sample indices
            batch_size: Batch size
            shuffle: Whether to shuffle within classes
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.class_indices = class_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get number of classes
        self.num_classes = len(class_indices)
        
        # Calculate samples per class in each batch
        samples_per_class = batch_size // self.num_classes
        remainder = batch_size % self.num_classes
        
        self.samples_per_class = [samples_per_class] * self.num_classes
        for i in range(remainder):
            self.samples_per_class[i] += 1
        
        # Calculate number of batches
        min_batches_per_class = [
            len(indices) // self.samples_per_class[i]
            for i, indices in enumerate(class_indices.values())
        ]
        self.num_batches = min(min_batches_per_class)
        
        # Calculate total size
        self.total_size = self.num_batches * batch_size
    
    def __iter__(self) -> Iterator[int]:
        """
        Create iterator over indices.
        
        Returns:
            Iterator over dataset indices
        """
        # Shuffle indices within each class if requested
        if self.shuffle:
            class_indices = {
                cls: indices.copy()
                for cls, indices in self.class_indices.items()
            }
            for indices in class_indices.values():
                random.shuffle(indices)
        else:
            class_indices = self.class_indices
        
        # Create batches
        for i in range(self.num_batches):
            batch_indices = []
            
            # Get indices for each class
            for cls, indices in class_indices.items():
                start_idx = i * self.samples_per_class[cls]
                end_idx = (i + 1) * self.samples_per_class[cls]
                batch_indices.extend(indices[start_idx:end_idx])
            
            # Shuffle within batch
            random.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        """
        Get sampler length.
        
        Returns:
            Total number of samples per epoch
        """
        return self.total_size