"""
Inference predictor for medical image segmentation.

This module provides a high-level interface for running inference
with trained segmentation models on medical images.
"""

import os
import time
from pathlib import Path
import glob
from typing import Dict, List, Optional, Union, Tuple, Any
import json

import numpy as np
import torch
import torch.nn as nn

from axon_ia.inference.sliding_window import SlidingWindowInference
from axon_ia.inference.postprocessing import apply_postprocessing
from axon_ia.utils.logger import get_logger
from axon_ia.utils.nifti_utils import load_nifti, save_nifti

logger = get_logger()


class Predictor:
    """
    High-level interface for model inference.
    
    This class handles the inference process for segmentation models,
    including preprocessing, sliding window inference, and postprocessing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sliding_window_size: Tuple[int, int, int] = (128, 128, 128),
        sliding_window_overlap: float = 0.5,
        sw_batch_size: int = 4,
        use_test_time_augmentation: bool = False,
        postprocessing_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Neural network model
            device: Device to use for inference
            sliding_window_size: Size of sliding window
            sliding_window_overlap: Overlap between adjacent windows
            sw_batch_size: Batch size for sliding window inference
            use_test_time_augmentation: Whether to use test-time augmentation
            postprocessing_params: Dictionary of postprocessing parameters
        """
        self.model = model
        self.device = device
        self.sliding_window_size = sliding_window_size
        self.sliding_window_overlap = sliding_window_overlap
        self.sw_batch_size = sw_batch_size
        self.use_test_time_augmentation = use_test_time_augmentation
        self.postprocessing_params = postprocessing_params or {}
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Create sliding window inference
        self.sliding_window_inference = SlidingWindowInference(
            roi_size=self.sliding_window_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.sliding_window_overlap,
            mode="gaussian",
            padding_mode="reflect",
            device=self.device,
            use_test_time_augmentation=self.use_test_time_augmentation
        )
    
    def predict_single_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        original_affine: Optional[np.ndarray] = None,
        apply_postproc: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (C, D, H, W)
            original_affine: Original affine matrix for NIfTI output
            apply_postproc: Whether to apply postprocessing
            
        Returns:
            Tuple of (binary prediction, probability map)
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            # Add batch dimension if missing
            if image.ndim == 3:
                image = image[np.newaxis, np.newaxis]  # (1, 1, D, H, W)
            elif image.ndim == 4:
                image = image[np.newaxis]  # (1, C, D, H, W)
                
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image).float()
        else:
            # Add batch dimension if missing
            if image.ndim == 3:
                image_tensor = image.unsqueeze(0).unsqueeze(0)
            elif image.ndim == 4:
                image_tensor = image.unsqueeze(0)
            else:
                image_tensor = image
        
        # Ensure float type
        image_tensor = image_tensor.float()
        
        # Measure inference time
        start_time = time.time()
        
        # Run inference with sliding window
        with torch.no_grad():
            prediction = self.sliding_window_inference(self.model, image_tensor)
        
        # Apply activation function based on output channels
        if prediction.shape[1] == 1:
            # Binary segmentation
            probabilities = torch.sigmoid(prediction)
        else:
            # Multi-class segmentation
            probabilities = torch.softmax(prediction, dim=1)
        
        # Convert to numpy
        prob_np = probabilities.detach().cpu().numpy()
        
        # Apply postprocessing if requested
        if apply_postproc:
            binary_pred = apply_postprocessing(
                prob_np, 
                **self.postprocessing_params
            )
        else:
            # Simple thresholding
            binary_pred = (prob_np > 0.5).astype(np.float32)
        
        # Log inference time
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.3f} seconds")
        
        return binary_pred, prob_np
    
    def predict_nifti_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        save_binary: bool = True,
        save_probabilities: bool = False,
        apply_postproc: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a NIfTI file.
        
        Args:
            input_path: Path to input NIfTI file
            output_dir: Directory to save output
            save_binary: Whether to save binary prediction
            save_probabilities: Whether to save probability map
            apply_postproc: Whether to apply postprocessing
            
        Returns:
            Dictionary with metadata
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load input image
        logger.info(f"Loading input image: {input_path}")
        image_data, meta = load_nifti(input_path, return_meta=True)
        
        # Add channel dimension if needed
        if image_data.ndim == 3:
            image_data = image_data[np.newaxis]  # (1, D, H, W)
        
        # Run inference
        logger.info("Running inference...")
        binary_pred, prob_map = self.predict_single_image(
            image=image_data,
            original_affine=meta["affine"],
            apply_postproc=apply_postproc
        )
        
        # Save outputs
        metadata = {
            "input_file": str(input_path),
            "prediction_time": time.time(),
            "image_shape": list(image_data.shape),
            "output_files": {}
        }
        
        # Save binary prediction
        if save_binary:
            # Generate output filename
            output_name = f"{input_path.stem}_pred{input_path.suffix}"
            output_path = output_dir / output_name
            
            # Save as NIfTI
            save_nifti(
                binary_pred[0, 0] if binary_pred.ndim >= 4 else binary_pred,
                output_path,
                affine=meta["affine"],
                header=meta["header"]
            )
            
            metadata["output_files"]["binary"] = str(output_path)
        
        # Save probability map
        if save_probabilities:
            # Generate output filename
            output_name = f"{input_path.stem}_prob{input_path.suffix}"
            output_path = output_dir / output_name
            
            # Save as NIfTI
            save_nifti(
                prob_map[0, 0] if prob_map.ndim >= 4 else prob_map,
                output_path,
                affine=meta["affine"],
                header=meta["header"]
            )
            
            metadata["output_files"]["probability"] = str(output_path)
        
        return metadata
    
    def predict_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        save_binary: bool = True,
        save_probabilities: bool = False,
        apply_postproc: bool = True,
        recursive: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run inference on a folder of NIfTI files.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save outputs
            file_pattern: Pattern to match input files
            save_binary: Whether to save binary predictions
            save_probabilities: Whether to save probability maps
            apply_postproc: Whether to apply postprocessing
            recursive: Whether to search recursively
            
        Returns:
            Dictionary with metadata for each file
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        if recursive:
            input_files = list(input_dir.glob(f"**/{file_pattern}"))
        else:
            input_files = list(input_dir.glob(file_pattern))
        
        logger.info(f"Found {len(input_files)} input files")
        
        # Process each file
        results = {}
        for input_file in input_files:
            logger.info(f"Processing: {input_file}")
            
            try:
                # Create relative path structure in output directory
                if recursive:
                    rel_path = input_file.relative_to(input_dir)
                    file_output_dir = output_dir / rel_path.parent
                else:
                    file_output_dir = output_dir
                
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run inference
                metadata = self.predict_nifti_file(
                    input_path=input_file,
                    output_dir=file_output_dir,
                    save_binary=save_binary,
                    save_probabilities=save_probabilities,
                    apply_postproc=apply_postproc
                )
                
                # Store results
                results[str(input_file.stem)] = metadata
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
        
        # Save metadata
        metadata_path = output_dir / "prediction_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=4)
        
        return results