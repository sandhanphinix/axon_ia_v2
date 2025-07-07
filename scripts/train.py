#!/usr/bin/env python
"""
Training script for Axon IA segmentation models.

This script handles the training process for all supported model architectures,
including data loading, model configuration, training loop, and evaluation.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Optional, Union, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.config import ConfigParser
from axon_ia.data import AxonDataset
from axon_ia.models import create_model
from axon_ia.losses import create_loss_function
from axon_ia.training import Trainer, create_scheduler
from axon_ia.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogging,
    WandBLogging
)
from axon_ia.utils.logger import get_logger
from axon_ia.utils.gpu_utils import get_device_info, select_best_device, optimize_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to config YAML file")
    
    parser.add_argument("--data-dir", type=str,
                        help="Path to data directory (overrides config)")
    
    parser.add_argument("--output-dir", type=str,
                        help="Path to output directory (overrides config)")
    
    parser.add_argument("--model", type=str,
                        help="Model architecture (overrides config)")
    
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint to resume from")
    
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs (overrides config)")
    
    parser.add_argument("--batch-size", type=int,
                        help="Batch size (overrides config)")
    
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate (overrides config)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--gpu", type=int,
                        help="GPU device ID to use")
    
    parser.add_argument("--amp", action="store_true", 
                        help="Use automatic mixed precision")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()


def setup_environment(args):
    """Set up training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = select_best_device()
    
    # Apply memory optimizations
    optimize_gpu_memory()
    
    # Create logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger(level=log_level)
    
    # Log system info
    logger.info(f"Using device: {device}")
    device_info = get_device_info()
    for dev, info in device_info.items():
        if dev.startswith("cuda"):
            logger.info(f"GPU {dev}: {info['name']}, {info['memory_total']:.2f} GB total")
    
    return device, logger


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    device, logger = setup_environment(args)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigParser(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.override("data.root_dir", args.data_dir)
    if args.output_dir:
        config.override("training.output_dir", args.output_dir)
    if args.model:
        config.override("model.architecture", args.model)
    if args.epochs:
        config.override("training.epochs", args.epochs)
    if args.batch_size:
        config.override("data.batch_size", args.batch_size)
    if args.learning_rate:
        config.override("optimizer.learning_rate", args.learning_rate)
    
    # Create output directory
    output_dir = Path(config.get("training.output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = output_dir / "config.yaml"
    config.save(config_save_path)
    
    # Create datasets
    logger.info("Creating datasets")
    data_config = config.get("data")
    
    train_dataset = AxonDataset(
        data_config["root_dir"],
        split="train",
        **data_config.get("dataset_params", {})
    )
    
    val_dataset = AxonDataset(
        data_config["root_dir"],
        split="val",
        **data_config.get("dataset_params", {})
    )
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get("batch_size", 4),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get("batch_size", 4),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model")
    model_config = config.get("model")
    model = create_model(
        architecture=model_config["architecture"],
        **model_config.get("params", {})
    )
    
    # Create loss function
    logger.info("Creating loss function")
    loss_config = config.get("loss")
    loss_fn = create_loss_function(
        loss_config["type"],
        **loss_config.get("params", {})
    )
    
    # Create optimizer
    logger.info("Creating optimizer")
    optimizer_config = config.get("optimizer")
    optimizer_type = optimizer_config.get("type", "adamw")
    lr = optimizer_config.get("learning_rate", 1e-4)
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=optimizer_config.get("weight_decay", 0.0)
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=optimizer_config.get("weight_decay", 0.01)
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=optimizer_config.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Create learning rate scheduler
    logger.info("Creating learning rate scheduler")
    scheduler_config = config.get("scheduler", {})
    scheduler = None
    if scheduler_config.get("use_scheduler", False):
        scheduler_type = scheduler_config.get("type", "cosine")
        scheduler = create_scheduler(
            optimizer,
            scheduler_type,
            num_epochs=config.get("training.epochs", 100),
            **scheduler_config.get("params", {})
        )
    
    # Create callbacks
    logger.info("Creating callbacks")
    callbacks = []
    
    # Early stopping callback
    if config.get("callbacks.early_stopping.enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=config.get("callbacks.early_stopping.monitor", "val_loss"),
                patience=config.get("callbacks.early_stopping.patience", 10),
                mode=config.get("callbacks.early_stopping.mode", "min")
            )
        )
    
    # Model checkpoint callback
    if config.get("callbacks.model_checkpoint.enabled", True):
        callbacks.append(
            ModelCheckpoint(
                filepath=output_dir / "checkpoints" / "model_{epoch:03d}.pth",
                monitor=config.get("callbacks.model_checkpoint.monitor", "val_loss"),
                save_best_only=config.get("callbacks.model_checkpoint.save_best_only", True),
                mode=config.get("callbacks.model_checkpoint.mode", "min")
            )
        )
    
    # TensorBoard callback
    if config.get("callbacks.tensorboard.enabled", True):
        callbacks.append(
            TensorBoardLogging(
                log_dir=output_dir / "tensorboard",
                log_freq=config.get("callbacks.tensorboard.log_freq", 1)
            )
        )
    
    # Weights & Biases callback
    if config.get("callbacks.wandb.enabled", False):
        callbacks.append(
            WandBLogging(
                project_name=config.get("callbacks.wandb.project_name", "axon_ia"),
                run_name=config.get("callbacks.wandb.run_name", None),
                config=config.to_dict()
            )
        )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        lr_scheduler=scheduler,
        callbacks=callbacks,
        grad_clip=config.get("training.grad_clip", None),
        use_amp=args.amp or config.get("training.use_amp", False),
        deep_supervision=model_config.get("params", {}).get("use_deep_supervision", False)
    )
    
    # Resume from checkpoint if specified
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint
        if not checkpoint_path and output_dir.exists():
            # Look for latest checkpoint
            checkpoints_dir = output_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("*.pth"))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer._load_checkpoint(checkpoint_path)
    
    # Train model
    logger.info("Starting training")
    training_config = config.get("training")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 100),
        val_interval=training_config.get("val_interval", 1),
        log_interval=training_config.get("log_interval", 10),
        checkpoint_dir=output_dir / "checkpoints",
        resume_from=args.checkpoint if args.resume else None,
        metrics=["dice", "iou"]
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "train": [
                    {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in epoch.items()}
                    for epoch in history["train"]
                ],
                "val": [
                    {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in epoch.items()}
                    for epoch in history["val"]
                ],
            },
            f,
            indent=4
        )
    
    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()