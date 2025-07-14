"""
Visualization utilities for medical image segmentation.

This module provides functions for visualizing segmentation results,
creating overlays, plotting metrics, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.colors as mcolors
from pathlib import Path


def plot_segmentation_overlay(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    alpha: float = 0.5,
    mask_color: str = 'red',
    pred_color: str = 'blue',
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    channel: int = 0
) -> plt.Figure:
    """
    Plot an image with segmentation mask and/or prediction overlay.
    
    Args:
        image: 3D or 4D image array. If 4D, first dimension is channels.
        mask: Ground truth mask (optional)
        prediction: Predicted mask (optional)
        slice_idx: Index of slice to plot (if None, uses middle slice)
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        alpha: Transparency of overlays
        mask_color: Color for ground truth mask
        pred_color: Color for prediction mask
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        channel: Channel to display for multi-channel images
        
    Returns:
        Matplotlib figure
    """
    # Handle multi-channel images - be more robust
    if image.ndim == 4:
        # Select the specified channel
        if channel < image.shape[0]:
            image = image[channel]
        else:
            # Default to first channel if requested channel doesn't exist
            image = image[0]
    elif image.ndim > 3:
        # Handle case where there are extra dimensions
        # Squeeze out singleton dimensions and take first channel
        image = np.squeeze(image)
        if image.ndim == 4:
            image = image[0]
    
    # Ensure we have a 3D image at this point
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after channel selection, got shape {image.shape}")
    
    # Default to middle slice if not specified
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    # Get slice based on axis
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :] if mask is not None else None
        pred_slice = prediction[slice_idx, :, :] if prediction is not None else None
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        mask_slice = mask[:, slice_idx, :] if mask is not None else None
        pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
    else:  # axis == 2
        img_slice = image[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx] if mask is not None else None
        pred_slice = prediction[:, :, slice_idx] if prediction is not None else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot image
    ax.imshow(img_slice, cmap='gray')
    
    # Plot mask overlay
    if mask_slice is not None:
        mask_rgba = np.zeros((*mask_slice.shape, 4))
        mask_color_rgb = mcolors.to_rgb(mask_color)
        mask_rgba[..., 0] = mask_color_rgb[0]
        mask_rgba[..., 1] = mask_color_rgb[1]
        mask_rgba[..., 2] = mask_color_rgb[2]
        mask_rgba[..., 3] = alpha * (mask_slice > 0)
        ax.imshow(mask_rgba)
    
    # Plot prediction overlay
    if pred_slice is not None:
        pred_rgba = np.zeros((*pred_slice.shape, 4))
        pred_color_rgb = mcolors.to_rgb(pred_color)
        pred_rgba[..., 0] = pred_color_rgb[0]
        pred_rgba[..., 1] = pred_color_rgb[1]
        pred_rgba[..., 2] = pred_color_rgb[2]
        pred_rgba[..., 3] = alpha * (pred_slice > 0)
        ax.imshow(pred_rgba)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        orientation = ['Sagittal', 'Coronal', 'Axial'][axis]
        ax.set_title(f"{orientation} Slice {slice_idx}")
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    if mask_slice is not None and pred_slice is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=mask_color, alpha=alpha, label='Ground Truth'),
            Patch(facecolor=pred_color, alpha=alpha, label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_metrics_per_patient(
    patient_metrics: Dict[str, Dict[str, float]],
    metric: str = "dice",
    sort_by: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot metrics for each patient as a bar chart.
    
    Args:
        patient_metrics: Dictionary of metrics per patient
        metric: Metric to plot
        sort_by: Metric to sort by (None for patient ID)
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Extract patient IDs and metric values
    patient_ids = list(patient_metrics.keys())
    values = [patient_metrics[pid].get(metric, np.nan) for pid in patient_ids]
    
    # Sort if requested
    if sort_by is not None:
        if sort_by == metric:
            # Sort by the metric being plotted
            sorted_indices = np.argsort(values)
            patient_ids = [patient_ids[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
        else:
            # Sort by another metric
            sort_values = [patient_metrics[pid].get(sort_by, np.nan) for pid in patient_ids]
            sorted_indices = np.argsort(sort_values)
            patient_ids = [patient_ids[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    bars = ax.bar(patient_ids, values)
    
    # Calculate statistics
    mean_val = np.nanmean(values)
    median_val = np.nanmedian(values)
    std_val = np.nanstd(values)
    
    # Add mean and median lines
    ax.axhline(mean_val, color='red', linestyle='-', label=f'Mean: {mean_val:.3f}')
    ax.axhline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric.capitalize()} per Patient (Mean: {mean_val:.3f}, Std: {std_val:.3f})")
    
    ax.set_xlabel("Patient ID")
    ax.set_ylabel(metric.capitalize())
    
    # Add legend
    ax.legend()
    
    # Rotate x-axis labels if many patients
    if len(patient_ids) > 10:
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_volume_correlation(
    true_volumes: Dict[str, float],
    pred_volumes: Dict[str, float],
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    units: str = 'mm³',
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot correlation between true and predicted volumes.
    
    Args:
        true_volumes: Dictionary of true volumes per patient
        pred_volumes: Dictionary of predicted volumes per patient
        figsize: Figure size
        title: Plot title
        units: Volume units
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    # Extract common patient IDs
    patient_ids = list(set(true_volumes.keys()) & set(pred_volumes.keys()))
    
    # Extract values
    true_vals = [true_volumes[pid] for pid in patient_ids]
    pred_vals = [pred_volumes[pid] for pid in patient_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter
    ax.scatter(true_vals, pred_vals, alpha=0.7)
    
    # Add identity line
    min_val = min(min(true_vals), min(pred_vals))
    max_val = max(max(true_vals), max(pred_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity')
    
    # Calculate correlation
    corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    
    # Calculate mean absolute error and relative error
    mae = np.mean(np.abs(np.array(true_vals) - np.array(pred_vals)))
    mre = np.mean(np.abs(np.array(true_vals) - np.array(pred_vals)) / np.array(true_vals))
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Volume Correlation (r={corr:.3f}, MAE={mae:.1f} {units})")
    
    ax.set_xlabel(f"True Volume ({units})")
    ax.set_ylabel(f"Predicted Volume ({units})")
    
    # Add metrics as text
    ax.text(0.05, 0.95, f"Correlation: {corr:.3f}\nMAE: {mae:.1f} {units}\nMRE: {mre:.3f}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_multiple_slices(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    axis: int = 2,
    num_slices: int = 3,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.5,
    mask_color: str = 'red',
    pred_color: str = 'blue',
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    channel: int = 0
) -> plt.Figure:
    """
    Plot multiple slices of an image with segmentation overlays.
    
    Args:
        image: 3D or 4D image array. If 4D, first dimension is channels.
        mask: Ground truth mask (optional)
        prediction: Predicted mask (optional)
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        num_slices: Number of slices to plot
        figsize: Figure size
        alpha: Transparency of overlays
        mask_color: Color for ground truth mask
        pred_color: Color for prediction mask
        title: Plot title
        save_path: Path to save the figure
        channel: Channel to display for multi-channel images
        
    Returns:
        Matplotlib figure
    """
    # Handle multi-channel images - be more robust  
    if image.ndim == 4:
        # Select the specified channel
        if channel < image.shape[0]:
            image = image[channel]
        else:
            # Default to first channel if requested channel doesn't exist
            image = image[0]
    elif image.ndim > 3:
        # Handle case where there are extra dimensions
        # Squeeze out singleton dimensions and take first channel
        image = np.squeeze(image)
        if image.ndim == 4:
            image = image[0]
    
    # Ensure we have a 3D image at this point
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image after channel selection, got shape {image.shape}")
    
    # Validate and process mask array
    if mask is not None:
        if mask.ndim == 4:
            # Take first channel for mask (segmentation masks are typically single channel)
            mask = mask[0] if mask.shape[0] == 1 else mask[0]
        elif mask.ndim > 3:
            mask = np.squeeze(mask)
            if mask.ndim == 4:
                mask = mask[0]
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask after processing, got shape {mask.shape}")
    
    # Validate and process prediction array
    if prediction is not None:
        if prediction.ndim == 4:
            # Take first channel for prediction (segmentation predictions are typically single channel)
            prediction = prediction[0] if prediction.shape[0] == 1 else prediction[0]
        elif prediction.ndim > 3:
            prediction = np.squeeze(prediction)
            if prediction.ndim == 4:
                prediction = prediction[0]
        if prediction.ndim != 3:
            raise ValueError(f"Expected 3D prediction after processing, got shape {prediction.shape}")
        
    # Determine slice indices
    total_slices = image.shape[axis]
    
    # Find slices with segmentation if available
    if mask is not None:
        # Find axial indices where the segmentation exists
        indices = []
        for i in range(total_slices):
            if axis == 0:
                slice_mask = mask[i, :, :]
            elif axis == 1:
                slice_mask = mask[:, i, :]
            else:  # axis == 2
                slice_mask = mask[:, :, i]
            
            if np.sum(slice_mask) > 0:
                indices.append(i)
        
        # If no segmentation found, use uniform spacing
        if not indices:
            indices = np.linspace(0, total_slices - 1, num_slices).astype(int).tolist()
        # If fewer slices with segmentation than requested, use all available
        elif len(indices) <= num_slices:
            pass
        # Otherwise, pick evenly spaced slices from those with segmentation
        else:
            step = max(1, len(indices) // num_slices)
            indices = indices[::step][:num_slices]
    else:
        # Without mask, use uniform spacing
        indices = np.linspace(0, total_slices - 1, num_slices).astype(int).tolist()
    
    # Create figure
    fig, axes = plt.subplots(1, len(indices), figsize=figsize)
    if len(indices) == 1:
        axes = [axes]
    
    # Plot each slice
    for i, slice_idx in enumerate(indices):
        # Get slices
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :] if mask is not None else None
            pred_slice = prediction[slice_idx, :, :] if prediction is not None else None
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :] if mask is not None else None
            pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
        else:  # axis == 2
            img_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx] if mask is not None else None
            pred_slice = prediction[:, :, slice_idx] if prediction is not None else None
        
        # Plot image
        axes[i].imshow(img_slice, cmap='gray')
        
        # Plot mask overlay
        if mask_slice is not None:
            mask_rgba = np.zeros((*mask_slice.shape, 4))
            mask_color_rgb = mcolors.to_rgb(mask_color)
            mask_rgba[..., 0] = mask_color_rgb[0]
            mask_rgba[..., 1] = mask_color_rgb[1]
            mask_rgba[..., 2] = mask_color_rgb[2]
            mask_rgba[..., 3] = alpha * (mask_slice > 0)
            axes[i].imshow(mask_rgba)
        
        # Plot prediction overlay
        if pred_slice is not None:
            pred_rgba = np.zeros((*pred_slice.shape, 4))
            pred_color_rgb = mcolors.to_rgb(pred_color)
            pred_rgba[..., 0] = pred_color_rgb[0]
            pred_rgba[..., 1] = pred_color_rgb[1]
            pred_rgba[..., 2] = pred_color_rgb[2]
            pred_rgba[..., 3] = alpha * (pred_slice > 0)
            axes[i].imshow(pred_rgba)
        
        # Set subtitle
        orientation = ['Sagittal', 'Coronal', 'Axial'][axis]
        axes[i].set_title(f"{orientation} Slice {slice_idx}")
        
        # Remove ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.85)
    
    # Add legend to the last subplot
    if mask is not None or prediction is not None:
        from matplotlib.patches import Patch
        legend_elements = []
        if mask is not None:
            legend_elements.append(Patch(facecolor=mask_color, alpha=alpha, label='Ground Truth'))
        if prediction is not None:
            legend_elements.append(Patch(facecolor=pred_color, alpha=alpha, label='Prediction'))
        
        axes[-1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig