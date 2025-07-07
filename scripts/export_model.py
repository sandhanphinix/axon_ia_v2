#!/usr/bin/env python
"""
Model export script for Axon IA.

This script exports trained PyTorch models to ONNX format
for efficient deployment in production environments.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.config import ConfigParser
from axon_ia.models import create_model
from axon_ia.utils.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint to export")
    
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save exported model")
    
    parser.add_argument("--input-shape", type=int, nargs="+",
                        default=[1, 4, 128, 128, 128],
                        help="Input tensor shape (batch_size, channels, depth, height, width)")
    
    parser.add_argument("--dynamic-axes", action="store_true",
                        help="Use dynamic axes for batch size")
    
    parser.add_argument("--optimize", action="store_true",
                        help="Apply ONNX optimizations")
    
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify the ONNX model")
    
    parser.add_argument("--test-export", action="store_true",
                        help="Test the exported model")
    
    return parser.parse_args()


def main():
    """Main model export function."""
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigParser(args.config)
    
    # Get input shape
    input_shape = tuple(args.input_shape)
    if len(input_shape) != 5:
        raise ValueError("Input shape must have 5 dimensions (B,C,D,H,W)")
    
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
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, requires_grad=False)
    
    # Define output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define dynamic axes if requested
    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export model to ONNX
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    start_time = time.time()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    export_time = time.time() - start_time
    logger.info(f"Export completed in {export_time:.2f} seconds")
    
    # Apply optimizations if requested
    if args.optimize or args.simplify:
        try:
            import onnx
            model = onnx.load(output_path)
            
            if args.optimize:
                logger.info("Applying ONNX optimizations")
                from onnxoptimizer import optimize
                
                # Common optimization passes
                passes = [
                    "eliminate_identity",
                    "eliminate_nop_dropout",
                    "eliminate_nop_pad",
                    "eliminate_unused_initializer",
                    "fuse_add_bias_into_conv",
                    "fuse_bn_into_conv"
                ]
                
                model = optimize(model, passes)
            
            if args.simplify:
                logger.info("Simplifying ONNX model")
                try:
                    from onnxsim import simplify
                    model, check = simplify(model)
                    if not check:
                        logger.warning("Simplified ONNX model could not be validated")
                except ImportError:
                    logger.warning("onnx-simplifier not installed, skipping simplification")
            
            # Save optimized model
            onnx.save(model, output_path)
            logger.info("Optimizations applied successfully")
            
        except ImportError:
            logger.warning("ONNX or optimization libraries not installed, skipping optimizations")
    
    # Test exported model if requested
    if args.test_export:
        try:
            import onnxruntime
            
            logger.info("Testing exported ONNX model")
            
            # Create inference session
            session = onnxruntime.InferenceSession(
                str(output_path), 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Run inference
            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            start_time = time.time()
            ort_outputs = session.run(None, ort_inputs)
            inference_time = time.time() - start_time
            
            logger.info(f"Test inference completed in {inference_time:.4f} seconds")
            logger.info(f"Output shape: {ort_outputs[0].shape}")
            
            # Compare with PyTorch model output
            with torch.no_grad():
                torch_output = model(dummy_input).numpy()
                
            mean_error = np.mean(np.abs(torch_output - ort_outputs[0]))
            max_error = np.max(np.abs(torch_output - ort_outputs[0]))
            
            logger.info(f"Mean absolute error: {mean_error:.6f}")
            logger.info(f"Max absolute error: {max_error:.6f}")
            
            if mean_error > 1e-4:
                logger.warning("Large discrepancy between PyTorch and ONNX outputs")
            else:
                logger.info("PyTorch and ONNX outputs match closely")
                
        except ImportError:
            logger.warning("ONNX Runtime not installed, skipping test")
    
    # Save model metadata
    metadata = {
        "model_architecture": model_config["architecture"],
        "model_params": model_config.get("params", {}),
        "input_shape": list(input_shape),
        "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "onnx_opset_version": 12,
        "dynamic_axes": args.dynamic_axes,
        "optimized": args.optimize,
        "simplified": args.simplify,
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model exported successfully to {output_path}")
    logger.info(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()