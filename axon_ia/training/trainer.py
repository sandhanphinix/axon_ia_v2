"""
Trainer class for training and evaluating neural networks.

This module provides a flexible Trainer class that handles
the training loop, evaluation, and various training utilities.
"""

import os
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Union, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from axon_ia.utils.logger import get_logger
from axon_ia.evaluation.metrics import dice_score, iou_score

logger = get_logger()


class Trainer:
    """
    Trainer class for neural networks.
    
    This class handles the training and validation loops, 
    metrics calculation, checkpointing, and more.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        lr_scheduler: Optional[object] = None,
        callbacks: Optional[List[object]] = None,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
        deep_supervision: bool = False,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimization algorithm
            loss_fn: Loss function
            device: Device to use for training
            lr_scheduler: Learning rate scheduler
            callbacks: List of callback objects
            grad_clip: Gradient clipping value
            use_amp: Whether to use automatic mixed precision
            deep_supervision: Whether the model uses deep supervision
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks or []
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.deep_supervision = deep_supervision
        
        # Training state
        self.current_epoch = 0
        self.current_iter = 0
        self.best_metric = float('inf')
        self.best_metric_name = None
        self.early_stop = False
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Move model to device
        self.model.to(self.device)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        val_interval: int = 1,
        log_interval: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        metrics: List[str] = ["dice"],
    ) -> Dict[str, List[Dict]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            val_interval: Validation interval in epochs
            log_interval: Logging interval in iterations
            checkpoint_dir: Directory to save checkpoints
            resume_from: Path to checkpoint to resume from
            metrics: List of metrics to calculate
            
        Returns:
            Training history
        """
        # Create checkpoint directory if provided
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if provided
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Initialize history
        history = {
            "train": [],
            "val": []
        }
        
        # Notify callbacks training is beginning
        for callback in self.callbacks:
            callback.on_training_begin(self)
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Notify callbacks epoch is beginning
            for callback in self.callbacks:
                callback.on_epoch_begin(self)
            
            # Train for one epoch
            train_logs = self._train_epoch(
                train_loader, 
                log_interval=log_interval
            )
            
            # Store training logs
            history["train"].append(train_logs)
            
            # Validate if needed
            val_logs = {}
            if val_interval > 0 and (epoch + 1) % val_interval == 0:
                val_logs = self._validate_epoch(val_loader, metrics)
                history["val"].append(val_logs)
            
            # Notify callbacks epoch is ending
            for callback in self.callbacks:
                callback.on_epoch_end(self, train_logs, val_logs)
            
            # Update learning rate scheduler if it's not a plateau scheduler
            if self.lr_scheduler is not None:
                if hasattr(self.lr_scheduler, 'step') and not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()
                elif isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_logs:
                    # For ReduceLROnPlateau, we need a validation metric
                    self.lr_scheduler.step(val_logs.get("val_loss", 0))
            
            # Log epoch results
            lr = self.optimizer.param_groups[0]['lr']
            log_message = f"Epoch {epoch + 1}/{epochs} - lr: {lr:.6f}"
            
            for k, v in train_logs.items():
                if isinstance(v, (int, float)):
                    log_message += f", {k}: {v:.4f}"
            
            for k, v in val_logs.items():
                if isinstance(v, (int, float)):
                    log_message += f", {k}: {v:.4f}"
            
            logger.info(log_message)
            
            # Save checkpoint if directory provided
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self._save_checkpoint(checkpoint_path)
            
            # Check for early stopping
            if self.early_stop:
                logger.info("Early stopping triggered, ending training")
                break
        
        # Notify callbacks training is ending
        for callback in self.callbacks:
            callback.on_training_end(self)
        
        return history
    
    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        log_interval: int = 10
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            log_interval: Logging interval in iterations
            
        Returns:
            Dict of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        running_loss = 0.0
        total_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()
        batch_start_time = time.time()
        
        # Iterate over batches
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            inputs = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)
            
            # Notify callbacks batch is beginning
            for callback in self.callbacks:
                callback.on_batch_begin(self)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self._calculate_loss(outputs, targets)
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward pass
                outputs = self.model(inputs)
                loss = self._calculate_loss(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if enabled
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update weights
                self.optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            loss_value = loss.item()
            running_loss += loss_value
            total_loss += loss_value * batch_size
            total_samples += batch_size
            
            # Update iteration counter
            self.current_iter += 1
            
            # Calculate batch metrics
            batch_metrics = {"loss": loss_value}
            
            # Log training progress
            if (batch_idx + 1) % log_interval == 0:
                # Calculate batch time and speed
                batch_time = time.time() - batch_start_time
                samples_per_sec = (log_interval * batch_size) / batch_time
                
                logger.info(
                    f"Epoch {self.current_epoch + 1} - Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"Loss: {running_loss / log_interval:.4f}, "
                    f"Samples/sec: {samples_per_sec:.2f}"
                )
                
                # Reset running loss and batch timer
                running_loss = 0.0
                batch_start_time = time.time()
            
            # Notify callbacks batch is ending
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_metrics)
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total_samples
        epoch_time = time.time() - epoch_start_time
        
        # Return metrics
        return {
            "loss": epoch_loss,
            "time": epoch_time,
            "lr": self.optimizer.param_groups[0]["lr"]
        }
    
    def _validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
        metrics: List[str] = ["dice"]
    ) -> Dict:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            metrics: List of metrics to calculate
            
        Returns:
            Dict of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        total_samples = 0
        metric_totals = {metric: 0.0 for metric in metrics}
        
        # Disable gradient calculation
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                inputs = batch["image"].to(self.device)
                targets = batch["mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, targets, is_validation=True)
                
                # Update metrics
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Calculate additional metrics
                # Get the main output if model returns multiple outputs
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                
                # Apply activation based on output channels
                if outputs.size(1) == 1:
                    preds = torch.sigmoid(outputs) > 0.5
                else:
                    preds = torch.argmax(outputs, dim=1, keepdim=True).float()
                
                # Calculate metrics for each sample in batch
                for metric in metrics:
                    for i in range(batch_size):
                        pred = preds[i:i+1]
                        target = targets[i:i+1]
                        
                        if metric == "dice":
                            score = dice_score(pred.cpu(), target.cpu())
                        elif metric == "iou":
                            score = iou_score(pred.cpu(), target.cpu())
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                        
                        metric_totals[metric] += score
        
        # Calculate average metrics
        val_loss = total_loss / total_samples
        metric_averages = {
            f"val_{metric}": metric_totals[metric] / total_samples
            for metric in metrics
        }
        
        # Return metrics
        results = {"val_loss": val_loss, **metric_averages}
        return results
    
    def _calculate_loss(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        targets: torch.Tensor,
        is_validation: bool = False
    ) -> torch.Tensor:
        """
        Calculate loss based on outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            is_validation: Whether this is a validation step
            
        Returns:
            Loss tensor
        """
        if self.deep_supervision and isinstance(outputs, (tuple, list)) and not is_validation:
            # For deep supervision, calculate loss for each output
            if hasattr(self.loss_fn, "deep_supervision_weights"):
                weights = self.loss_fn.deep_supervision_weights
            else:
                # Default weights
                weights = [0.5, 0.25, 0.125, 0.0625][:len(outputs)]
                # Normalize weights to sum to 1
                weights = [w / sum(weights) for w in weights]
            
            # Calculate weighted sum of losses
            loss = sum(w * self.loss_fn(out, targets) for w, out in zip(weights, outputs))
        else:
            # For single output or validation, calculate loss for main output
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            loss = self.loss_fn(outputs, targets)
        
        return loss
    
    def _save_checkpoint(
        self,
        path: Union[str, Path],
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "iteration": self.current_iter,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name
        }
        
        # Add learning rate scheduler state if exists
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        # Add AMP scaler state if exists
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        if is_best:
            # Create best checkpoint path
            best_path = path.parent / "best_model.pth"
            # Copy checkpoint to best path
            import shutil
            shutil.copy(path, best_path)
    
    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        self.current_iter = checkpoint.get("iteration", 0)
        self.best_metric = checkpoint.get("best_metric", float('inf'))
        self.best_metric_name = checkpoint.get("best_metric_name", None)
        
        # Load learning rate scheduler state if exists
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")
        
        # Load AMP scaler state if exists
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scaler state: {e}")
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch - 1}")