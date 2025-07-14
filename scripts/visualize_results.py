#!/usr/bin/env python
"""
Comprehensive visualization script for model analysis.

This script creates detailed visualizations for understanding model performance,
including prediction overlays, error analysis, and statistical summaries.
"""

import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
from scipy import ndimage
from sklearn.metrics import confusion_matrix

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from axon_ia.config import ConfigParser
from axon_ia.utils.nifti_utils import load_nifti


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate comprehensive model visualizations")
    
    parser.add_argument("--predictions-dir", type=str, required=True,
                        help="Directory containing model predictions")
    
    parser.add_argument("--ground-truth-dir", type=str, required=True,
                        help="Directory containing ground truth masks")
    
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing original images")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save visualizations")
    
    parser.add_argument("--metrics-file", type=str,
                        help="Path to metrics JSON file")
    
    parser.add_argument("--modality", type=str, default="flair",
                        choices=["flair", "b0", "b1000", "t2star"],
                        help="Modality to use for background visualization")
    
    parser.add_argument("--slice-selection", type=str, default="center",
                        choices=["center", "max_lesion", "all"],
                        help="How to select slices for visualization")
    
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to visualize")
    
    parser.add_argument("--split", type=str, default="test",
                        help="Data split (train/val/test)")
    
    return parser.parse_args()


def get_bounding_box(mask: np.ndarray, padding: int = 5) -> Tuple[slice, slice, slice]:
    """
    Get bounding box around non-zero regions in mask.
    
    Args:
        mask: 3D binary mask
        padding: Padding around bounding box
        
    Returns:
        Tuple of slices for each dimension
    """
    # Find non-zero coordinates
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # No lesions found, return center crop
        h, w, d = mask.shape
        return (
            slice(h//4, 3*h//4),
            slice(w//4, 3*w//4),
            slice(d//4, 3*d//4)
        )
    
    # Get bounding box
    min_h, max_h = coords[0].min(), coords[0].max()
    min_w, max_w = coords[1].min(), coords[1].max()
    min_d, max_d = coords[2].min(), coords[2].max()
    
    # Ensure minimum size for each dimension
    min_size = 10  # Minimum size in voxels
    
    # Expand if too small
    if max_h - min_h < min_size:
        center_h = (min_h + max_h) // 2
        min_h = max(0, center_h - min_size // 2)
        max_h = min(h, center_h + min_size // 2)
    
    if max_w - min_w < min_size:
        center_w = (min_w + max_w) // 2
        min_w = max(0, center_w - min_size // 2)
        max_w = min(w, center_w + min_size // 2)
    
    if max_d - min_d < min_size:
        center_d = (min_d + max_d) // 2
        min_d = max(0, center_d - min_size // 2)
        max_d = min(d, center_d + min_size // 2)
    
    # Add padding
    h, w, d = mask.shape
    min_h = max(0, min_h - padding)
    max_h = min(h, max_h + padding)
    min_w = max(0, min_w - padding)
    max_w = min(w, max_w + padding)
    min_d = max(0, min_d - padding)
    max_d = min(d, max_d + padding)
    
    return slice(min_h, max_h), slice(min_w, max_w), slice(min_d, max_d)


def create_overlay_visualization(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    slice_idx: int,
    axis: int = 2,
    title: str = ""
) -> plt.Figure:
    """
    Create overlay visualization comparing ground truth and prediction.
    
    Args:
        image: Background image
        ground_truth: Ground truth mask
        prediction: Predicted mask
        slice_idx: Slice index to visualize
        axis: Axis along which to slice
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    # Validate input shapes
    if any(dim == 0 for dim in image.shape):
        raise ValueError(f"Invalid image shape: {image.shape}")
    if any(dim == 0 for dim in ground_truth.shape):
        raise ValueError(f"Invalid ground_truth shape: {ground_truth.shape}")
    if any(dim == 0 for dim in prediction.shape):
        raise ValueError(f"Invalid prediction shape: {prediction.shape}")
    
    # Validate slice index
    max_slice = image.shape[axis] - 1
    if slice_idx > max_slice:
        slice_idx = max_slice
    if slice_idx < 0:
        slice_idx = 0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Get slices
    if axis == 0:  # Sagittal
        img_slice = image[slice_idx, :, :]
        gt_slice = ground_truth[slice_idx, :, :]
        pred_slice = prediction[slice_idx, :, :]
    elif axis == 1:  # Coronal
        img_slice = image[:, slice_idx, :]
        gt_slice = ground_truth[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :]
    else:  # Axial
        img_slice = image[:, :, slice_idx]
        gt_slice = ground_truth[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
    
    # Validate slice shapes
    if any(dim == 0 for dim in img_slice.shape):
        raise ValueError(f"Invalid image slice shape: {img_slice.shape}")
    if any(dim == 0 for dim in gt_slice.shape):
        raise ValueError(f"Invalid ground truth slice shape: {gt_slice.shape}")
    if any(dim == 0 for dim in pred_slice.shape):
        raise ValueError(f"Invalid prediction slice shape: {pred_slice.shape}")
    
    # Original image
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth overlay
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.5)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Prediction overlay
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(pred_slice, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')
    
    # Comparison (GT=Red, Pred=Blue, Overlap=Purple)
    # Create RGB overlay
    overlay = np.zeros((*img_slice.shape, 3))
    overlay[:, :, 0] = gt_slice  # Red channel for GT
    overlay[:, :, 2] = pred_slice  # Blue channel for prediction
    
    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].imshow(overlay, alpha=0.6)
    axes[1, 1].set_title('Comparison (GT=Red, Pred=Blue, Overlap=Purple)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def create_error_analysis_plot(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    sample_id: str
) -> plt.Figure:
    """
    Create error analysis visualization.
    
    Args:
        ground_truth: Ground truth mask
        prediction: Predicted mask
        sample_id: Sample identifier
        
    Returns:
        Matplotlib figure
    """
    # Convert to binary
    gt_binary = (ground_truth > 0.5).astype(np.float32)
    pred_binary = (prediction > 0.5).astype(np.float32)
    
    # Calculate error maps
    true_positive = gt_binary * pred_binary
    false_positive = (1 - gt_binary) * pred_binary
    false_negative = gt_binary * (1 - pred_binary)
    true_negative = (1 - gt_binary) * (1 - pred_binary)
    
    # Create error map
    error_map = np.zeros_like(gt_binary)
    error_map[true_positive > 0] = 1  # TP = 1
    error_map[false_positive > 0] = 2  # FP = 2
    error_map[false_negative > 0] = 3  # FN = 3
    
    # Find slice with most errors
    error_counts = np.sum(error_map > 1, axis=(0, 1))
    best_slice = np.argmax(error_counts)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Error Analysis - {sample_id}', fontsize=16)
    
    # Color map for errors
    colors = ['black', 'green', 'red', 'orange']  # Background, TP, FP, FN
    cmap = ListedColormap(colors)
    
    # Error map
    axes[0, 0].imshow(error_map[:, :, best_slice], cmap=cmap, vmin=0, vmax=3)
    axes[0, 0].set_title('Error Map (Green=TP, Red=FP, Orange=FN)')
    axes[0, 0].axis('off')
    
    # Volume comparison
    gt_volume = np.sum(gt_binary)
    pred_volume = np.sum(pred_binary)
    volumes = [gt_volume, pred_volume]
    labels = ['Ground Truth', 'Prediction']
    
    axes[0, 1].bar(labels, volumes, color=['red', 'blue'], alpha=0.7)
    axes[0, 1].set_title('Volume Comparison (voxels)')
    axes[0, 1].set_ylabel('Number of voxels')
    
    # Confusion matrix
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    cm = confusion_matrix(gt_flat, pred_flat)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Error statistics
    tp = np.sum(true_positive)
    fp = np.sum(false_positive)
    fn = np.sum(false_negative)
    tn = np.sum(true_negative)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    stats_text = f"""
    True Positives: {tp:.0f}
    False Positives: {fp:.0f}
    False Negatives: {fn:.0f}
    True Negatives: {tn:.0f}
    
    Precision: {precision:.3f}
    Recall: {recall:.3f}
    Dice Score: {dice:.3f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics')
    
    plt.tight_layout()
    return fig


def create_metrics_summary_plot(metrics_data: Dict) -> plt.Figure:
    """
    Create summary plot of all metrics.
    
    Args:
        metrics_data: Dictionary containing metrics for all patients
        
    Returns:
        Matplotlib figure
    """
    # Extract per-patient metrics
    per_patient = metrics_data.get('per_patient', {})
    
    if not per_patient:
        # Create dummy plot if no data
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No per-patient metrics available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Metrics Summary')
        return fig
    
    # Organize metrics
    patients = list(per_patient.keys())
    metrics = list(per_patient[patients[0]].keys())
    
    # Create subplot for each metric
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Per-Patient Metrics Distribution', fontsize=16)
    
    for i, metric in enumerate(metrics):
        values = [per_patient[patient][metric] for patient in patients]
        
        # Box plot
        axes[i].boxplot(values, labels=[metric])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        
        # Add mean line
        mean_val = np.mean(values)
        axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                       label=f'Mean: {mean_val:.3f}')
        axes[i].legend()
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main visualization function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics if available
    metrics_data = {}
    if args.metrics_file and Path(args.metrics_file).exists():
        with open(args.metrics_file, 'r') as f:
            metrics_data = json.load(f)
    
    # Get list of prediction files
    pred_dir = Path(args.predictions_dir)
    pred_files = list(pred_dir.glob("*_pred.nii.gz"))
    
    # Limit number of samples
    if len(pred_files) > args.num_samples:
        pred_files = pred_files[:args.num_samples]
    
    print(f"Processing {len(pred_files)} samples...")
    
    # Process each sample
    for pred_file in pred_files:
        sample_id = pred_file.name.replace("_pred.nii.gz", "")
        print(f"Processing {sample_id}...")
        
        # Load prediction
        prediction = load_nifti(pred_file)
        
        # Remove extra dimensions from prediction (e.g., channel dimension)
        prediction = np.squeeze(prediction)
        
        # Load ground truth - handle hierarchical directory structure
        gt_file = Path(args.ground_truth_dir) / args.split / sample_id / "perfroi.nii.gz"
        if not gt_file.exists():
            # Fallback to flat structure
            gt_file = Path(args.ground_truth_dir) / f"{sample_id}_perfroi.nii.gz"
            if not gt_file.exists():
                print(f"Ground truth not found for {sample_id}, skipping...")
                continue
        ground_truth = load_nifti(gt_file)
        
        # Load image (use specified modality) - handle hierarchical directory structure
        img_file = Path(args.images_dir) / args.split / sample_id / f"{args.modality}.nii.gz"
        if not img_file.exists():
            # Fallback to flat structure
            img_file = Path(args.images_dir) / f"{sample_id}_{args.modality}.nii.gz"
            if not img_file.exists():
                print(f"Image not found for {sample_id}, skipping...")
                continue
        image = load_nifti(img_file)
        
        # Apply same preprocessing as dataset (resize to standard size)
        from monai.transforms import Resize
        target_size = (128, 128, 128)  # Standard size used in dataset
        resize_transform = Resize(spatial_size=target_size, mode="trilinear")
        
        # Convert to torch tensors and handle dimensions properly for MONAI
        import torch
        
        # Handle different input dimensions
        if image.ndim == 3:
            # 3D image, add channel dimension
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        elif image.ndim == 4:
            # 4D image, use as is
            image_tensor = torch.from_numpy(image).float()
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
        
        if ground_truth.ndim == 3:
            # 3D ground truth, add channel dimension
            gt_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0)
        elif ground_truth.ndim == 4:
            # 4D ground truth, use as is
            gt_tensor = torch.from_numpy(ground_truth).float()
        else:
            raise ValueError(f"Unexpected ground truth dimensions: {ground_truth.ndim}")
        
        # Apply resize transform
        image_resized = resize_transform(image_tensor)
        ground_truth_resized = resize_transform(gt_tensor)
        
        # Convert back to numpy and remove channel dimension if single channel
        if image_resized.shape[0] == 1:
            image = image_resized.squeeze(0).numpy()
        else:
            image = image_resized[0].numpy()  # Take first channel
            
        if ground_truth_resized.shape[0] == 1:
            ground_truth = ground_truth_resized.squeeze(0).numpy()
        else:
            ground_truth = ground_truth_resized[0].numpy()  # Take first channel
        
        # Validate shapes are consistent
        if image.shape != ground_truth.shape:
            print(f"Shape mismatch for {sample_id}: image {image.shape} vs ground_truth {ground_truth.shape}, skipping...")
            continue
        
        if prediction.shape != ground_truth.shape:
            print(f"Shape mismatch for {sample_id}: prediction {prediction.shape} vs ground_truth {ground_truth.shape}, skipping...")
            continue
        
        # Get bounding box around lesions
        bbox = get_bounding_box(ground_truth)
        
        # Crop to region of interest
        image_crop = image[bbox]
        gt_crop = ground_truth[bbox]
        pred_crop = prediction[bbox]
        
        # Validate cropped shapes
        if any(dim == 0 for dim in image_crop.shape):
            print(f"Invalid crop for {sample_id}: {image_crop.shape}, skipping...")
            continue
        
        # Select slice for visualization
        if args.slice_selection == "center":
            slice_idx = image_crop.shape[2] // 2
        elif args.slice_selection == "max_lesion":
            # Find slice with maximum lesion area
            lesion_areas = np.sum(gt_crop, axis=(0, 1))
            slice_idx = np.argmax(lesion_areas)
        
        # Create overlay visualization
        overlay_fig = create_overlay_visualization(
            image_crop, gt_crop, pred_crop, slice_idx,
            title=f"Sample: {sample_id} (Slice {slice_idx})"
        )
        
        # Save overlay
        overlay_path = output_dir / f"{sample_id}_overlay.png"
        overlay_fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close(overlay_fig)
        
        # Create error analysis
        error_fig = create_error_analysis_plot(ground_truth, prediction, sample_id)
        
        # Save error analysis
        error_path = output_dir / f"{sample_id}_error_analysis.png"
        error_fig.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close(error_fig)
    
    # Create metrics summary if available
    if metrics_data:
        summary_fig = create_metrics_summary_plot(metrics_data)
        summary_path = output_dir / "metrics_summary.png"
        summary_fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(summary_fig)
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
