"""
Visualization utilities for medical image data.

This module provides functions for visualizing medical images,
segmentations, and training metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
from pathlib import Path
import io

import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting


def plot_slices(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    axis: int = 2,
    slices: Optional[List[int]] = None,
    n_slices: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    alpha: float = 0.3,
    show_colorbar: bool = True,
    mask_color: str = 'red',
    pred_color: str = 'blue',
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Plot slices from a 3D image with optional mask and prediction overlays.
    
    Args:
        img: 3D image array [Z, Y, X] or [C, Z, Y, X]
        mask: Optional ground truth segmentation mask
        pred: Optional prediction segmentation mask
        axis: Axis along which to slice (0=z, 1=y, 2=x)
        slices: List of slice indices to plot, or None for auto-selection
        n_slices: Number of slices to plot if slices is None
        figsize: Figure size
        alpha: Transparency of the overlays
        show_colorbar: Whether to show colorbar
        mask_color: Color for ground truth mask
        pred_color: Color for prediction mask
        title: Optional title for the figure
        vmin: Minimum value for image intensity scaling
        vmax: Maximum value for image intensity scaling
        
    Returns:
        Matplotlib figure
    """
    # Handle channel dimension if present
    if img.ndim == 4:
        if img.shape[0] > 3:
            # Too many channels, just take the first one
            img = img[0]
        else:
            # Try to create RGB composite if 3 channels
            img = np.stack([img[i] for i in range(min(3, img.shape[0]))], axis=-1)
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Get image dimensions
    dims = img.shape
    
    # Determine slice indices
    if slices is None:
        # Auto-select slices
        total_slices = dims[axis]
        if total_slices <= n_slices:
            slices = list(range(total_slices))
        else:
            start = total_slices // (n_slices + 1)
            end = total_slices - start
            slices = np.linspace(start, end, n_slices, dtype=int).tolist()
    
    # Create figure
    n_cols = min(5, len(slices))
    n_rows = (len(slices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine slice function based on axis
    def get_slice(data, i, ax):
        if ax == 0:
            return data[i]
        elif ax == 1:
            return data[:, i]
        else:  # ax == 2
            return data[:, :, i]
    
    # Plot each slice
    for i, slice_idx in enumerate(slices):
        if i >= len(axes):
            break
            
        # Extract slice
        img_slice = get_slice(img, slice_idx, axis)
        
        # Plot image
        im = axes[i].imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Add mask overlay if provided
        if mask is not None:
            mask_slice = get_slice(mask, slice_idx, axis)
            axes[i].imshow(mask_slice > 0, alpha=alpha * (mask_slice > 0), cmap=f'{mask_color}_alpha')
        
        # Add prediction overlay if provided
        if pred is not None:
            pred_slice = get_slice(pred, slice_idx, axis)
            axes[i].imshow(pred_slice > 0, alpha=alpha * (pred_slice > 0), cmap=f'{pred_color}_alpha')
        
        # Set title and remove ticks
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Hide unused axes
    for i in range(len(slices), len(axes)):
        axes[i].axis('off')
    
    # Add colorbar
    if show_colorbar:
        fig.colorbar(im, ax=axes, shrink=0.6)
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 10),
    smoothing: float = 0.0,
) -> plt.Figure:
    """
    Plot training curves from metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics, where keys are metric names and values are lists of values
        figsize: Figure size
        smoothing: Exponential moving average smoothing factor (0 = no smoothing)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        # Apply smoothing if requested
        if smoothing > 0:
            smoothed = []
            last = values[0]
            for v in values:
                last = last * smoothing + v * (1 - smoothing)
                smoothed.append(last)
            values = smoothed
        
        # Plot
        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('Epoch' if i == len(metrics) - 1 else '')
        axes[i].grid(True)
    
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        figsize: Figure size
        normalize: Whether to normalize by row
        title: Optional title for the figure
        
    Returns:
        Matplotlib figure
    """
    # Normalize if requested
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    # Add title
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    return fig


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Convert matplotlib figure to numpy array.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        RGB image as numpy array
    """
    # Draw the figure to a buffer
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    
    # Convert buffer to numpy array
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close(fig)
    
    return img_arr


def save_plot_grid(
    save_path: Union[str, Path],
    rows: List[List[np.ndarray]],
    titles: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = None,
    cmap: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Save a grid of images to a file.
    
    Args:
        save_path: Path to save the image
        rows: List of rows, where each row is a list of images
        titles: Optional list of column titles
        row_labels: Optional list of row labels
        figsize: Figure size
        cmap: Colormap
        vmin: Minimum value for intensity scaling
        vmax: Maximum value for intensity scaling
    """
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Add column titles
    if titles is not None:
        for i, title in enumerate(titles):
            if i < n_cols:
                axes[0, i].set_title(title)
    
    # Add row labels
    if row_labels is not None:
        for i, label in enumerate(row_labels):
            if i < n_rows:
                axes[i, 0].set_ylabel(label, fontsize=12)
    
    # Plot images
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            if j < n_cols:
                axes[i, j].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    
    # Hide unused axes
    for i in range(n_rows):
        for j in range(len(rows[i]), n_cols):
            axes[i, j].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def generate_volume_visualization(
    volume: Union[np.ndarray, torch.Tensor],
    segmentation: Optional[Union[np.ndarray, torch.Tensor]] = None,
    prediction: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_slices_per_axis: int = 5,
    figsize: Tuple[int, int] = (15, 15),
) -> plt.Figure:
    """
    Generate a comprehensive visualization of a 3D volume with segmentations.
    
    Args:
        volume: 3D volume array [Z, Y, X] or [1, Z, Y, X]
        segmentation: Optional ground truth segmentation mask
        prediction: Optional prediction segmentation mask
        num_slices_per_axis: Number of slices to show per axis
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Remove singleton dimensions
    volume = np.squeeze(volume)
    if segmentation is not None:
        segmentation = np.squeeze(segmentation)
    if prediction is not None:
        prediction = np.squeeze(prediction)
    
    # Check dimensions
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot axial slices (top row)
    for i, z in enumerate(np.linspace(0, volume.shape[0]-1, num_slices_per_axis, dtype=int)):
        ax = fig.add_subplot(3, num_slices_per_axis, i+1)
        ax.imshow(volume[z], cmap='gray')
        
        if segmentation is not None:
            ax.imshow(segmentation[z], alpha=0.3, cmap='red')
        if prediction is not None:
            ax.imshow(prediction[z], alpha=0.3, cmap='blue')
        
        ax.set_title(f"Axial {z}")
        ax.axis('off')
    
    # Plot coronal slices (middle row)
    for i, y in enumerate(np.linspace(0, volume.shape[1]-1, num_slices_per_axis, dtype=int)):
        ax = fig.add_subplot(3, num_slices_per_axis, i+1+num_slices_per_axis)
        ax.imshow(volume[:, y, :], cmap='gray')
        
        if segmentation is not None:
            ax.imshow(segmentation[:, y, :], alpha=0.3, cmap='red')
        if prediction is not None:
            ax.imshow(prediction[:, y, :], alpha=0.3, cmap='blue')
        
        ax.set_title(f"Coronal {y}")
        ax.axis('off')
    
    # Plot sagittal slices (bottom row)
    for i, x in enumerate(np.linspace(0, volume.shape[2]-1, num_slices_per_axis, dtype=int)):
        ax = fig.add_subplot(3, num_slices_per_axis, i+1+2*num_slices_per_axis)
        ax.imshow(volume[:, :, x], cmap='gray')
        
        if segmentation is not None:
            ax.imshow(segmentation[:, :, x], alpha=0.3, cmap='red')
        if prediction is not None:
            ax.imshow(prediction[:, :, x], alpha=0.3, cmap='blue')
        
        ax.set_title(f"Sagittal {x}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig