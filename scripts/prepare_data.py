#!/usr/bin/env python
"""
Data preparation script for Axon IA.

This script preprocesses raw medical imaging data into a standardized
format ready for training and evaluation.
"""

import os
import sys
import argparse
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional, Union, Tuple, Any
import time
import random

import numpy as np
from tqdm import tqdm

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.utils.logger import get_logger
from axon_ia.utils.nifti_utils import load_nifti, save_nifti
from axon_ia.data.preprocessing import (
    resample_to_spacing,
    normalize_intensity,
    crop_foreground,
    standardize_orientation
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for training")
    
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing raw data")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save processed data")
    
    parser.add_argument("--config", type=str,
                        help="Path to config YAML file with preprocessing parameters")
    
    parser.add_argument("--modalities", type=str, nargs="+", 
                        default=["flair", "t1", "t2", "dwi"],
                        help="List of modalities to process")
    
    parser.add_argument("--target", type=str, default="mask",
                        help="Name of the segmentation target")
    
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target spacing in mm")
    
    parser.add_argument("--normalize", type=str, default="z_score",
                        choices=["z_score", "percentile", "min_max"],
                        help="Intensity normalization method")
    
    parser.add_argument("--crop-foreground", action="store_true",
                        help="Crop to foreground region")
    
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Dataset splits to create")
    
    parser.add_argument("--split-ratio", type=float, nargs="+", default=[0.7, 0.15, 0.15],
                        help="Ratio for train/val/test split")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting")
    
    return parser.parse_args()


def find_case_files(
    case_dir: Path,
    modalities: List[str],
    target: str,
    file_ext: str = ".nii.gz"
) -> Tuple[Dict[str, Path], Optional[Path]]:
    """
    Find modality and target files for a case.
    
    Args:
        case_dir: Case directory
        modalities: List of modalities to look for
        target: Target segmentation name
        file_ext: File extension to match
        
    Returns:
        Tuple of (modality_files, target_file)
    """
    modality_files = {}
    target_file = None
    
    # Look for files matching modalities
    for modality in modalities:
        # Try exact match
        exact_match = case_dir / f"{modality}{file_ext}"
        if exact_match.exists():
            modality_files[modality] = exact_match
            continue
        
        # Try case-insensitive match
        for file_path in case_dir.glob(f"*{file_ext}"):
            if modality.lower() in file_path.stem.lower():
                modality_files[modality] = file_path
                break
    
    # Look for target file
    target_exact = case_dir / f"{target}{file_ext}"
    if target_exact.exists():
        target_file = target_exact
    else:
        # Try case-insensitive match
        for file_path in case_dir.glob(f"*{file_ext}"):
            if target.lower() in file_path.stem.lower() or "mask" in file_path.stem.lower():
                target_file = file_path
                break
    
    return modality_files, target_file


def preprocess_case(
    case_id: str,
    modality_files: Dict[str, Path],
    target_file: Optional[Path],
    output_dir: Path,
    modalities: List[str],
    target: str,
    spacing: List[float],
    normalize_method: str,
    crop_foreground_flag: bool
) -> Dict[str, Any]:
    """
    Preprocess a single case.
    
    Args:
        case_id: Case identifier
        modality_files: Dictionary of modality files
        target_file: Target segmentation file
        output_dir: Output directory
        modalities: List of modalities
        target: Target segmentation name
        spacing: Target spacing in mm
        normalize_method: Intensity normalization method
        crop_foreground_flag: Whether to crop to foreground
        
    Returns:
        Dictionary with preprocessing metadata
    """
    # Create output case directory
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Load modality images
    modality_data = {}
    modality_meta = {}
    
    for modality in modalities:
        if modality in modality_files:
            data, meta = load_nifti(modality_files[modality], return_meta=True)
            modality_data[modality] = data
            modality_meta[modality] = meta
    
    # Load target if available
    target_data = None
    target_meta = None
    if target_file:
        target_data, target_meta = load_nifti(target_file, return_meta=True)
    
    # Standardize orientation
    for modality in modality_data:
        modality_data[modality] = standardize_orientation(modality_data[modality])
    
    if target_data is not None:
        target_data = standardize_orientation(target_data)
    
    # Resample to target spacing
    for modality in modality_data:
        original_spacing = modality_meta[modality]["zooms"]
        modality_data[modality] = resample_to_spacing(
            modality_data[modality],
            original_spacing=original_spacing,
            target_spacing=spacing,
            interpolation="linear",
            is_mask=False
        )
    
    if target_data is not None:
        original_spacing = target_meta["zooms"]
        target_data = resample_to_spacing(
            target_data,
            original_spacing=original_spacing,
            target_spacing=spacing,
            interpolation="nearest",
            is_mask=True
        )
    
    # Crop to foreground if requested
    if crop_foreground_flag:
        # Use FLAIR or first available modality for foreground detection
        reference_modality = "flair" if "flair" in modality_data else list(modality_data.keys())[0]
        
        # Create a mask from the reference modality
        mask = modality_data[reference_modality] > 0
        
        # Crop all modalities based on this mask
        for modality in modality_data:
            modality_data[modality], _, crop_indices = crop_foreground(
                modality_data[modality],
                mask,
                margin=10
            )
        
        # Crop target if available
        if target_data is not None:
            target_data = target_data[
                crop_indices[0][0]:crop_indices[0][1],
                crop_indices[1][0]:crop_indices[1][1],
                crop_indices[2][0]:crop_indices[2][1]
            ]
    
    # Normalize intensity
    for modality in modality_data:
        modality_data[modality] = normalize_intensity(
            modality_data[modality],
            mode=normalize_method
        )
    
    # Ensure target is binary
    if target_data is not None:
        target_data = (target_data > 0.5).astype(np.float32)
    
    # Save preprocessed files
    metadata = {"case_id": case_id, "preprocessing": {}}
    
    for modality in modality_data:
        output_path = case_dir / f"{modality}.nii.gz"
        save_nifti(modality_data[modality], output_path)
        metadata["preprocessing"][modality] = {
            "original_file": str(modality_files[modality]),
            "output_file": str(output_path),
            "shape": modality_data[modality].shape
        }
    
    if target_data is not None:
        output_path = case_dir / f"{target}.nii.gz"
        save_nifti(target_data, output_path)
        metadata["preprocessing"]["target"] = {
            "original_file": str(target_file),
            "output_file": str(output_path),
            "shape": target_data.shape
        }
    
    return metadata


def main():
    """Main data preparation function."""
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if provided
    config = {}
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    modalities = config.get("modalities", args.modalities)
    target = config.get("target", args.target)
    spacing = config.get("spacing", args.spacing)
    normalize_method = config.get("normalize_mode", args.normalize)
    crop_foreground_flag = config.get("crop_foreground", args.crop_foreground)
    
    # Find all case directories
    input_dir = Path(args.input_dir)
    case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(case_dirs)} case directories")
    
    # Check for valid cases
    valid_cases = []
    for case_dir in case_dirs:
        modality_files, target_file = find_case_files(case_dir, modalities, target)
        
        # Check if all required modalities are present
        all_modalities_present = all(mod in modality_files for mod in modalities)
        
        # Case is valid if it has all modalities and the target
        if all_modalities_present and target_file is not None:
            valid_cases.append({
                "case_id": case_dir.name,
                "case_dir": case_dir,
                "modality_files": modality_files,
                "target_file": target_file
            })
    
    logger.info(f"Found {len(valid_cases)} valid cases with all modalities and target")
    
    # Split cases into train/val/test
    random.shuffle(valid_cases)
    split_ratios = args.split_ratio
    
    # Ensure split ratios sum to 1
    split_ratios = np.array(split_ratios)
    split_ratios = split_ratios / split_ratios.sum()
    
    # Calculate split sizes
    num_cases = len(valid_cases)
    split_sizes = [int(np.round(ratio * num_cases)) for ratio in split_ratios]
    
    # Adjust last split size to ensure all cases are used
    split_sizes[-1] = num_cases - sum(split_sizes[:-1])
    
    # Create splits
    split_cases = {}
    start_idx = 0
    for i, split in enumerate(args.splits):
        end_idx = start_idx + split_sizes[i]
        split_cases[split] = valid_cases[start_idx:end_idx]
        start_idx = end_idx
    
    # Create split directories
    for split in args.splits:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
    
    # Process cases for each split
    metadata = {"splits": {}}
    
    for split, cases in split_cases.items():
        logger.info(f"Processing {len(cases)} cases for {split} split")
        
        split_metadata = []
        for case in tqdm(cases, desc=f"Processing {split}"):
            try:
                # Preprocess case
                case_metadata = preprocess_case(
                    case_id=case["case_id"],
                    modality_files=case["modality_files"],
                    target_file=case["target_file"],
                    output_dir=output_dir / split,
                    modalities=modalities,
                    target=target,
                    spacing=spacing,
                    normalize_method=normalize_method,
                    crop_foreground_flag=crop_foreground_flag
                )
                split_metadata.append(case_metadata)
            except Exception as e:
                logger.error(f"Error processing case {case['case_id']}: {e}")
        
        metadata["splits"][split] = {
            "num_cases": len(split_metadata),
            "cases": split_metadata
        }
    
    # Add metadata about preprocessing parameters
    metadata["preprocessing_params"] = {
        "modalities": modalities,
        "target": target,
        "spacing": spacing,
        "normalize_method": normalize_method,
        "crop_foreground": crop_foreground_flag
    }
    
    # Save metadata
    metadata_path = output_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Preprocessing complete. Results saved to {output_dir}")
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()