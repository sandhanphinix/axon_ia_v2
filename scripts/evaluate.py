#!/usr/bin/env python
"""
Evaluation script for Axon IA segmentation models.

This script evaluates trained models on validation or test data,
calculating metrics and generating visualizations.
"""

import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.config import ConfigParser
from axon_ia.data import AxonDataset
from axon_ia.models import create_model
from axon_ia.evaluation import compute_metrics, generate_evaluation_report
from axon_ia.utils.logger import get_logger
from axon_ia.utils.gpu_utils import get_device_info, select_best_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model")
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint to evaluate")
    
    parser.add_argument("--data-dir", type=str,
                        help="Path to data directory (overrides config)")
    
    parser.add_argument("--output-dir", type=str,
                        help="Path to output directory (overrides config)")
    
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    
    parser.add_argument("--gpu", type=int,
                        help="GPU device ID to use")
    
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save model predictions")
    
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate evaluation report")
    
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["dice", "iou", "hausdorff", "precision", "recall"],
                        help="Metrics to calculate")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = select_best_device()
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigParser(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.override("data.root_dir", args.data_dir)
    
    # Create output directory
    if args.output_dir:
        # Use the provided output directory directly
        eval_dir = Path(args.output_dir)
    else:
        # Use config output directory with evaluation subdirectory
        output_dir = Path(config.get("training.output_dir"))
        eval_dir = output_dir / f"evaluation_{args.split}"
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    logger.info(f"Creating {args.split} dataset")
    data_config = config.get("data")
    
    # For evaluation, we don't need training transforms
    dataset_params = data_config.get("dataset_params", {}).copy()
    if "transform" in dataset_params:
        # Remove transform for evaluation to use defaults (no augmentation)
        del dataset_params["transform"]
    
    dataset = AxonDataset(
        data_config["root_dir"],
        split=args.split,
        modalities=data_config["modalities"],
        target=data_config["target"],
        **dataset_params
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
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
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Run evaluation
    logger.info(f"Evaluating model on {args.split} dataset")
    
    # Store predictions and targets for metrics calculation
    all_predictions = []
    all_targets = []
    all_images = []
    all_sample_ids = []
    
    # Metrics for each sample
    patient_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get data
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            sample_ids = batch["sample_id"]
            
            # Forward pass
            predictions = model(images)
            
            # If deep supervision, take the first output
            if isinstance(predictions, (tuple, list)):
                predictions = predictions[0]
            
            # Apply sigmoid for binary segmentation
            if predictions.size(1) == 1:
                predictions = torch.sigmoid(predictions)
                binary_preds = predictions > 0.5
            else:
                # Apply softmax for multi-class
                predictions = torch.softmax(predictions, dim=1)
                binary_preds = torch.argmax(predictions, dim=1, keepdim=True).float()
            
            # Calculate metrics for each sample
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                sample_pred = binary_preds[i:i+1].detach().cpu()
                sample_target = targets[i:i+1].detach().cpu()
                
                # Compute metrics
                metrics = compute_metrics(sample_pred, sample_target, args.metrics)
                patient_metrics[sample_id] = metrics
            
            # Store for overall metrics calculation
            all_predictions.append(binary_preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Store images for visualization if needed
            if args.generate_report:
                all_images.append(images.detach().cpu())
                all_sample_ids.extend(sample_ids)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate overall metrics
    logger.info("Calculating overall metrics")
    overall_metrics = compute_metrics(all_predictions, all_targets, args.metrics)
    
    # Print overall metrics
    for metric, value in overall_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save metrics to JSON
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "overall": {k: float(v) for k, v in overall_metrics.items()},
            "per_patient": {
                pid: {k: float(v) for k, v in metrics.items()}
                for pid, metrics in patient_metrics.items()
            }
        }, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate evaluation report if requested
    if args.generate_report:
        logger.info("Generating evaluation report")
        
        # Concatenate all images
        all_images = torch.cat(all_images, dim=0)
        
        # Create dictionaries for visualization
        patient_images = {}
        patient_targets = {}
        patient_predictions = {}
        
        for i, sample_id in enumerate(all_sample_ids):
            patient_images[sample_id] = all_images[i].numpy()
            patient_targets[sample_id] = all_targets[i].numpy()
            patient_predictions[sample_id] = all_predictions[i].numpy()
        
        # Generate report
        report_path = generate_evaluation_report(
            patient_metrics=patient_metrics,
            output_dir=eval_dir,
            patient_images=patient_images,
            patient_targets=patient_targets,
            patient_predictions=patient_predictions,
            title=f"Evaluation Report - {args.split.capitalize()} Dataset"
        )
        
        logger.info(f"Evaluation report generated at {report_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        logger.info("Saving predictions")
        
        # Create predictions directory
        pred_dir = eval_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        import nibabel as nib
        from axon_ia.utils.nifti_utils import save_nifti
        
        for i, sample_id in enumerate(all_sample_ids):
            # Get prediction for this sample
            pred = all_predictions[i].numpy()
            
            # Convert boolean predictions to uint8 for NIfTI compatibility
            if pred.dtype == bool:
                pred = pred.astype(np.uint8)
            
            # Save as NIfTI
            pred_path = pred_dir / f"{sample_id}_pred.nii.gz"
            save_nifti(pred, pred_path)
        
        logger.info(f"Predictions saved to {pred_dir}")
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()