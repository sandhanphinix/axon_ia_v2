"""
Learning rate schedulers for training neural networks.

This module provides various learning rate schedulers
and a factory function to create them from configuration.
"""

from typing import Dict, Any, Optional, Union, List
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    This scheduler implements a cosine annealing schedule with
    an optional linear warmup period at the beginning.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get updated learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
                                  (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler.
    
    This scheduler decreases the learning rate polynomially
    with an optional linear warmup period.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        power: float = 0.9,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            max_epochs: Total number of epochs
            power: Power for polynomial decay
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.max_epochs = max_epochs
        self.power = power
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get updated learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            factor = (1 - (self.last_epoch - self.warmup_epochs) / 
                     (self.max_epochs - self.warmup_epochs)) ** self.power
            return [self.eta_min + (base_lr - self.eta_min) * factor
                    for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler to be composed with other schedulers.
    
    This scheduler implements only the linear warmup phase and
    can be used with other PyTorch schedulers for the main schedule.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            last_epoch: The index of last epoch
            verbose: Whether to print message on each update
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """Get updated learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_epochs: int,
    **kwargs
) -> _LRScheduler:
    """
    Create a learning rate scheduler based on type and parameters.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - int(kwargs.get("warmup_epochs", 0)),
            eta_min=float(kwargs.get("min_lr", 0))
        )
    
    elif scheduler_type == "cosine_warmup":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(kwargs.get("warmup_epochs", 5)),
            max_epochs=num_epochs,
            warmup_start_lr=float(kwargs.get("warmup_start_lr", 1e-6)),
            eta_min=float(kwargs.get("min_lr", 0))
        )
    
    elif scheduler_type == "polynomial":
        return PolynomialLRScheduler(
            optimizer,
            max_epochs=num_epochs,
            power=float(kwargs.get("power", 0.9)),
            warmup_epochs=int(kwargs.get("warmup_epochs", 0)),
            warmup_start_lr=float(kwargs.get("warmup_start_lr", 1e-6)),
            eta_min=float(kwargs.get("min_lr", 0))
        )
    
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1)
        )
    
    elif scheduler_type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get("milestones", [30, 60, 90]),
            gamma=kwargs.get("gamma", 0.1)
        )
    
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            threshold=kwargs.get("threshold", 1e-4),
            threshold_mode=kwargs.get("threshold_mode", "rel"),
            min_lr=kwargs.get("min_lr", 0)
        )
    
    elif scheduler_type == "warmup_with_scheduler":
        # Create base scheduler
        base_scheduler_type = kwargs.get("base_scheduler", "cosine")
        base_kwargs = kwargs.copy()
        
        # Exclude warmup parameters from base scheduler
        base_scheduler = create_scheduler(optimizer, base_scheduler_type, num_epochs, **base_kwargs)
        
        # Create warmup wrapper
        warmup_scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-6)
        )
        
        # Create sequential scheduler
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, base_scheduler],
            milestones=[kwargs.get("warmup_epochs", 5)]
        )
    
    else:
        available_schedulers = ["cosine", "cosine_warmup", "polynomial", 
                                "step", "multistep", "plateau", "warmup_with_scheduler"]
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {available_schedulers}")