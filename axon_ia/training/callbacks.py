"""
Training callbacks for monitoring and improving the training process.

This module provides a collection of callback classes for monitoring training,
saving checkpoints, visualizing metrics, and more.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import numpy as np

from axon_ia.utils.logger import get_logger

logger = get_logger()


class Callback:
    """Base callback class with default implementations."""
    
    def on_training_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    def on_training_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, train_logs=None, val_logs=None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_logs=None):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting.
    
    This callback stops training when a monitored metric has
    stopped improving for a specified number of epochs.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which to stop
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' for minimizing or maximizing the metric
            baseline: Baseline value for the monitored metric
            restore_best_weights: Whether to restore the best model weights
        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        # Initialize
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == "min":
            self.best = float("inf")
            self.monitor_op = lambda x, y: x < y - min_delta
        elif mode == "max":
            self.best = float("-inf")
            self.monitor_op = lambda x, y: x > y + min_delta
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
    
    def on_training_begin(self, trainer):
        """Called at the beginning of training."""
        # Initialize best based on baseline if provided
        if self.baseline is not None:
            if self.monitor_op(self.baseline, self.best):
                self.best = self.baseline
    
    def on_epoch_end(self, trainer, train_logs=None, val_logs=None):
        """Called at the end of each epoch to check for early stopping."""
        # Skip if validation metrics aren't available
        if not val_logs or self.monitor not in val_logs:
            return
        
        # Get current value of monitored metric
        current = val_logs[self.monitor]
        
        # Store best weights if current model is best
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    logger.info("Restoring model weights from the end of the best epoch")
                    trainer.model.load_state_dict(self.best_weights)
    
    def on_training_end(self, trainer):
        """Called at the end of training."""
        if self.stopped_epoch > 0:
            logger.info(f"Early stopping triggered at epoch {self.stopped_epoch + 1}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    This callback saves model checkpoints at specified intervals
    or when a monitored metric improves.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "val_loss",
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        save_freq: Union[str, int] = "epoch",
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path pattern to save checkpoints (can include {epoch} placeholder)
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model based on monitored metric
            save_weights_only: Whether to save only weights or full model
            mode: 'min' or 'max' for minimizing or maximizing the metric
            save_freq: Saving frequency ('epoch' or number of batches)
            verbose: Whether to print information when saving checkpoints
        """
        super().__init__()
        
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Create directory
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize
        if mode == "min":
            self.best = float("inf")
            self.monitor_op = lambda x, y: x < y
        elif mode == "max":
            self.best = float("-inf")
            self.monitor_op = lambda x, y: x > y
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
        
        self.batches_since_last_save = 0
    
    def on_epoch_end(self, trainer, train_logs=None, val_logs=None):
        """Called at the end of each epoch to save checkpoint."""
        if self.save_freq == "epoch":
            self._save_checkpoint(trainer, train_logs, val_logs)
    
    def on_batch_end(self, trainer, batch_logs=None):
        """Called at the end of each batch to save checkpoint if needed."""
        if isinstance(self.save_freq, int):
            self.batches_since_last_save += 1
            if self.batches_since_last_save >= self.save_freq:
                self.batches_since_last_save = 0
                self._save_checkpoint(trainer, None, None)
    
    def _save_checkpoint(self, trainer, train_logs=None, val_logs=None):
        """
        Save checkpoint if conditions are met.
        
        Args:
            trainer: Trainer instance
            train_logs: Training logs
            val_logs: Validation logs
        """
        # Format filepath with current epoch
        formatted_filepath = str(self.filepath).format(epoch=trainer.current_epoch + 1)
        
        # Check if we should save based on monitored metric
        if self.save_best_only:
            # Skip if validation metrics aren't available
            if not val_logs or self.monitor not in val_logs:
                return
            
            # Get current value of monitored metric
            current = val_logs[self.monitor]
            
            # Save only if monitored metric improved
            if self.monitor_op(current, self.best):
                if self.verbose:
                    message = f"{self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {formatted_filepath}"
                    logger.info(message)
                
                self.best = current
                
                # Save model
                if self.save_weights_only:
                    torch.save(trainer.model.state_dict(), formatted_filepath)
                else:
                    state_dict = {
                        "model_state_dict": trainer.model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "epoch": trainer.current_epoch,
                        "best_metric": self.best,
                        "best_metric_name": self.monitor,
                    }
                    
                    # Add LR scheduler if present
                    if trainer.lr_scheduler is not None:
                        state_dict["lr_scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
                    
                    torch.save(state_dict, formatted_filepath)
        else:
            # Always save
            if self.verbose:
                logger.info(f"Saving model to {formatted_filepath}")
            
            # Save model
            if self.save_weights_only:
                torch.save(trainer.model.state_dict(), formatted_filepath)
            else:
                state_dict = {
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "epoch": trainer.current_epoch,
                }
                
                # Add LR scheduler if present
                if trainer.lr_scheduler is not None:
                    state_dict["lr_scheduler_state_dict"] = trainer.lr_scheduler.state_dict()
                
                torch.save(state_dict, formatted_filepath)


class TensorBoardLogging(Callback):
    """
    TensorBoard logging callback.
    
    This callback logs metrics, learning rate, and optionally
    model architecture to TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_freq: int = 10,
        log_images: bool = False,
        log_graph: bool = False,
        max_images: int = 3
    ):
        """
        Initialize TensorBoard logging callback.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            log_freq: Batch logging frequency
            log_images: Whether to log sample images
            log_graph: Whether to log model graph
            max_images: Maximum number of images to log
        """
        super().__init__()
        
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.log_images = log_images
        self.log_graph = log_graph
        self.max_images = max_images
        
        # Create TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            logger.warning("TensorBoard not available. TensorBoard logging disabled.")
            self.enabled = False
    
    def on_training_begin(self, trainer):
        """Called at the beginning of training to log model graph."""
        if not self.enabled:
            return
        
        # Log model graph if requested
        if self.log_graph:
            try:
                # Get a sample input
                for batch in trainer.train_loader:
                    inputs = batch["image"][:1].to(trainer.device)
                    self.writer.add_graph(trainer.model, inputs)
                    break
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def on_epoch_end(self, trainer, train_logs=None, val_logs=None):
        """Called at the end of each epoch to log metrics."""
        if not self.enabled:
            return
        
        # Log training metrics
        if train_logs:
            for key, value in train_logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, trainer.current_epoch)
        
        # Log validation metrics
        if val_logs:
            for key, value in val_logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, trainer.current_epoch)
        
        # Log learning rate
        self.writer.add_scalar(
            "lr",
            trainer.optimizer.param_groups[0]["lr"],
            trainer.current_epoch
        )
        
        # Log sample images and predictions if requested
        if self.log_images and hasattr(trainer, "val_loader"):
            self._log_images(trainer)
    
    def on_batch_end(self, trainer, batch_logs=None):
        """Called at the end of each batch to log metrics."""
        if not self.enabled or batch_logs is None:
            return
        
        # Log batch metrics at specified frequency
        if trainer.current_iter % self.log_freq == 0:
            for key, value in batch_logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"batch/{key}", value, trainer.current_iter)
    
    def _log_images(self, trainer):
        """
        Log sample images and predictions to TensorBoard.
        
        Args:
            trainer: Trainer instance
        """
        try:
            # Get a batch of validation data
            for batch in trainer.val_loader:
                # Move data to device
                inputs = batch["image"][:self.max_images].to(trainer.device)
                targets = batch["mask"][:self.max_images].to(trainer.device)
                
                # Get predictions
                trainer.model.eval()
                with torch.no_grad():
                    outputs = trainer.model(inputs)
                    
                    # If model returns tuple/list, take first output
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                
                # Apply activation function based on task
                if outputs.size(1) == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                
                # Move data back to CPU for logging
                inputs = inputs.cpu()
                targets = targets.cpu()
                preds = preds.cpu()
                
                # Log images
                for i in range(min(inputs.size(0), self.max_images)):
                    # Log input image (take first channel if multi-channel)
                    input_img = inputs[i, 0] if inputs.size(1) > 1 else inputs[i, 0]
                    self.writer.add_image(
                        f"input/sample_{i}",
                        input_img.unsqueeze(0),  # Add channel dim
                        trainer.current_epoch
                    )
                    
                    # Log target mask
                    target_img = targets[i, 0] if targets.size(1) > 1 else targets[i, 0]
                    self.writer.add_image(
                        f"target/sample_{i}",
                        target_img.unsqueeze(0),  # Add channel dim
                        trainer.current_epoch
                    )
                    
                    # Log prediction
                    pred_img = preds[i, 0] if preds.size(1) > 1 else preds[i, 0]
                    self.writer.add_image(
                        f"prediction/sample_{i}",
                        pred_img.unsqueeze(0),  # Add channel dim
                        trainer.current_epoch
                    )
                
                break
        except Exception as e:
            logger.warning(f"Failed to log images: {e}")
    
    def on_training_end(self, trainer):
        """Called at the end of training to close the writer."""
        if self.enabled:
            self.writer.close()


class WandBLogging(Callback):
    """
    Weights & Biases logging callback.
    
    This callback logs metrics, hyperparameters, model architecture,
    and optionally images to Weights & Biases.
    """
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_freq: int = 10,
        log_images: bool = False,
        max_images: int = 3
    ):
        """
        Initialize W&B logging callback.
        
        Args:
            project_name: W&B project name
            run_name: W&B run name (default: auto-generated)
            config: Configuration dictionary for hyperparameters
            log_freq: Batch logging frequency
            log_images: Whether to log sample images
            max_images: Maximum number of images to log
        """
        super().__init__()
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.log_freq = log_freq
        self.log_images = log_images
        self.max_images = max_images
        
        # Check if wandb is available
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            logger.warning("Weights & Biases not available. W&B logging disabled.")
            self.enabled = False
    
    def on_training_begin(self, trainer):
        """Called at the beginning of training to initialize W&B."""
        if not self.enabled:
            return
        
        # Initialize W&B run
        try:
            self.wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                reinit=True
            )
            
            # Log model architecture
            try:
                self.wandb.watch(trainer.model, log="all")
            except Exception as e:
                logger.warning(f"Failed to log model architecture: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False
    
    def on_epoch_end(self, trainer, train_logs=None, val_logs=None):
        """Called at the end of each epoch to log metrics."""
        if not self.enabled:
            return
        
        # Combine training and validation logs
        log_dict = {}
        
        if train_logs:
            for key, value in train_logs.items():
                if isinstance(value, (int, float)):
                    log_dict[f"train/{key}"] = value
        
        if val_logs:
            for key, value in val_logs.items():
                if isinstance(value, (int, float)):
                    log_dict[f"val/{key}"] = value
        
        # Add learning rate
        log_dict["lr"] = trainer.optimizer.param_groups[0]["lr"]
        
        # Add epoch
        log_dict["epoch"] = trainer.current_epoch + 1
        
        # Log metrics
        try:
            self.wandb.log(log_dict)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")
        
        # Log sample images and predictions if requested
        if self.log_images and hasattr(trainer, "val_loader"):
            self._log_images(trainer)
    
    def on_batch_end(self, trainer, batch_logs=None):
        """Called at the end of each batch to log metrics."""
        if not self.enabled or batch_logs is None:
            return
        
        # Log batch metrics at specified frequency
        if trainer.current_iter % self.log_freq == 0:
            log_dict = {}
            for key, value in batch_logs.items():
                if isinstance(value, (int, float)):
                    log_dict[f"batch/{key}"] = value
            
            # Add step
            log_dict["step"] = trainer.current_iter
            
            # Log metrics
            try:
                self.wandb.log(log_dict)
            except Exception as e:
                logger.warning(f"Failed to log batch metrics to W&B: {e}")
    
    def _log_images(self, trainer):
        """
        Log sample images and predictions to W&B.
        
        Args:
            trainer: Trainer instance
        """
        try:
            # Get a batch of validation data
            for batch in trainer.val_loader:
                # Move data to device
                inputs = batch["image"][:self.max_images].to(trainer.device)
                targets = batch["mask"][:self.max_images].to(trainer.device)
                
                # Get predictions
                trainer.model.eval()
                with torch.no_grad():
                    outputs = trainer.model(inputs)
                    
                    # If model returns tuple/list, take first output
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                
                # Apply activation function based on task
                if outputs.size(1) == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                
                # Move data back to CPU
                inputs = inputs.cpu().numpy()
                targets = targets.cpu().numpy()
                preds = preds.cpu().numpy()
                
                # Log images
                for i in range(min(inputs.shape[0], self.max_images)):
                    # Extract 2D slices from middle of volume
                    input_slice = inputs[i, 0, inputs.shape[2]//2]
                    target_slice = targets[i, 0, targets.shape[2]//2]
                    pred_slice = preds[i, 0, preds.shape[2]//2]
                    
                    # Log to W&B
                    try:
                        self.wandb.log({
                            f"images/sample_{i}": self.wandb.Image(
                                input_slice,
                                caption=f"Input Sample {i}"
                            ),
                            f"targets/sample_{i}": self.wandb.Image(
                                target_slice,
                                caption=f"Target Sample {i}"
                            ),
                            f"predictions/sample_{i}": self.wandb.Image(
                                pred_slice,
                                caption=f"Prediction Sample {i}"
                            )
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log image to W&B: {e}")
                
                break
        except Exception as e:
            logger.warning(f"Failed to log images to W&B: {e}")
    
    def on_training_end(self, trainer):
        """Called at the end of training to finalize W&B run."""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")