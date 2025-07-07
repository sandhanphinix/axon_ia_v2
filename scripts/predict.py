#!/usr/bin/env python
"""
Prediction script for Axon IA segmentation models.

This script applies trained models to new data for inference,
handling preprocessing, prediction, and postprocessing.
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
from axon_ia.models import create_model
from axon_ia.inference import Predictor
from axon_ia.utils.logger import get_logger
from axon_ia.utils.gpu_utils import select_best_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint to use")
    
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input images")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save predictions")
    
    parser.add_argument("--file-pattern", type=str, default="*.nii.gz",
                        help="Pattern to match input files")
    
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    
    parser.add_argument("--gpu", type=int,
                        help="GPU device ID to use")
    
    parser.add_argument("--save-probability", action="store_true",
                        help="Save probability maps in addition to binary segmentations")
    
    parser.add_argument("--use-tta", action="store_true",
                        help="Use test-time augmentation")
    
    return parser.parse_args()


def main():
    """Main prediction function."""
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
    
    # Get inference config
    inference_config = config.get("inference", {})
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create predictor
    logger.info("Creating predictor")
    predictor = Predictor(
        model=model,
        device=device,
        sliding_window_size=inference_config.get("sliding_window_size", (128, 128, 128)),
        sliding_window_overlap=inference_config.get("overlap", 0.5),
        sw_batch_size=args.batch_size,
        use_test_time_augmentation=args.use_tta or inference_config.get("use_test_time_augmentation", False),
        postprocessing_params=inference_config.get("postprocessing", {}),
    )
    
    # Run prediction on directory
    logger.info(f"Running prediction on {args.input_dir}")
    metadata = predictor.predict_folder(
        input_dir=args.input_dir,
        output_dir=output_dir,
        file_pattern=args.file_pattern,
        save_binary=True,
        save_probabilities=args.save_probability,
    )
    
    # Save metadata
    metadata_path = output_dir / "prediction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {k: {kk: str(vv) if isinstance(vv, Path) else vv 
                for kk, vv in v.items()}
             for k, v in metadata.items()},
            f,
            indent=4
        )
    
    logger.info(f"Predictions saved to {output_dir}")
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()