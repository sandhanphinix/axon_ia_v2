"""
Evaluation metrics for medical image segmentation.

This module provides a collection of evaluation metrics specifically
designed for assessing medical image segmentation results.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Callable, Tuple
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import precision_score as sklearn_precision
from sklearn.metrics import recall_score as sklearn_recall
from sklearn.metrics import f1_score as sklearn_f1


def dice_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate Dice similarity coefficient.
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_true > 0.5).astype(np.float32)
    
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calculate Dice
    intersection = np.sum(y_pred * y_true)
    dice = (2.0 * intersection + smooth) / (np.sum(y_pred) + np.sum(y_true) + smooth)
    
    return float(dice)


def iou_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate Intersection over Union (IoU) / Jaccard index.
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_true > 0.5).astype(np.float32)
    
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calculate IoU
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def precision_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate precision (positive predictive value).
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Precision score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_true > 0.5).astype(np.float32)
    
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calculate precision
    intersection = np.sum(y_pred * y_true)
    precision = (intersection + smooth) / (np.sum(y_pred) + smooth)
    
    return float(precision)


def recall_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate recall (sensitivity).
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Recall score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_true > 0.5).astype(np.float32)
    
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calculate recall
    intersection = np.sum(y_pred * y_true)
    recall = (intersection + smooth) / (np.sum(y_true) + smooth)
    
    return float(recall)


def specificity_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate specificity.
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Specificity score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.float32)
    y_true = (y_true > 0.5).astype(np.float32)
    
    # Flatten arrays
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calculate specificity
    true_neg = np.sum((1 - y_pred) * (1 - y_true))
    false_pos = np.sum(y_pred * (1 - y_true))
    specificity = (true_neg + smooth) / (true_neg + false_pos + smooth)
    
    return float(specificity)


def f1_score(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-5
) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        F1 score
    """
    # F1 is equivalent to Dice for binary segmentation
    return dice_score(y_pred, y_true, smooth)


def hausdorff_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    percentile: Optional[float] = 95,
    spacing: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Calculate Hausdorff distance between two binary volumes.
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        percentile: Percentile for computing distance (None for max)
        spacing: Voxel spacing in mm
        
    Returns:
        Hausdorff distance
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)
    
    # Handle empty arrays
    if np.sum(y_pred) == 0 or np.sum(y_true) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distance from prediction to ground truth
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
        
        # Calculate distance transforms
        if spacing is not None:
            dt_true = distance_transform_edt(1 - y_true, sampling=spacing)
            dt_pred = distance_transform_edt(1 - y_pred, sampling=spacing)
        else:
            dt_true = distance_transform_edt(1 - y_true)
            dt_pred = distance_transform_edt(1 - y_pred)
        
        # Get surface points
        pred_surface = ((y_pred - binary_erosion(y_pred, iterations=1)) > 0)
        true_surface = ((y_true - binary_erosion(y_true, iterations=1)) > 0)
        
        # Compute directed Hausdorff distances
        dist_pred_to_true = dt_true[pred_surface]
        dist_true_to_pred = dt_pred[true_surface]
        
        if percentile is not None:
            # Percentile Hausdorff distance
            hausdorff_pred_to_true = np.percentile(dist_pred_to_true, percentile)
            hausdorff_true_to_pred = np.percentile(dist_true_to_pred, percentile)
        else:
            # Maximum Hausdorff distance
            hausdorff_pred_to_true = np.max(dist_pred_to_true)
            hausdorff_true_to_pred = np.max(dist_true_to_pred)
        
        # Symmetric Hausdorff distance
        hausdorff = max(hausdorff_pred_to_true, hausdorff_true_to_pred)
        
        return float(hausdorff)
    
    except Exception as e:
        # Fallback to simplified version
        from scipy.spatial.distance import directed_hausdorff
        
        # Get coordinates of points in each set
        true_points = np.argwhere(y_true > 0)
        pred_points = np.argwhere(y_pred > 0)
        
        # Check for empty arrays
        if len(true_points) == 0 or len(pred_points) == 0:
            return float('inf')
        
        # Ensure both arrays have the same number of dimensions
        if true_points.shape[1] != pred_points.shape[1]:
            # This should not happen with proper input, but handle it gracefully
            return float('inf')
        
        # Apply spacing if provided
        if spacing is not None:
            spacing_array = np.array(spacing)
            if len(spacing_array) == true_points.shape[1]:
                true_points = true_points * spacing_array
                pred_points = pred_points * spacing_array
        
        # Calculate directed Hausdorff distances
        try:
            hausdorff_pred_to_true = directed_hausdorff(pred_points, true_points)[0]
            hausdorff_true_to_pred = directed_hausdorff(true_points, pred_points)[0]
            
            # Symmetric Hausdorff distance
            hausdorff = max(hausdorff_pred_to_true, hausdorff_true_to_pred)
            
            return float(hausdorff)
        except Exception as inner_e:
            # If Hausdorff calculation still fails, return infinity
            return float('inf')


def surface_dice(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    tolerance: float = 1.0,
    spacing: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Calculate surface Dice coefficient (boundary Dice).
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        tolerance: Distance tolerance in mm or voxels
        spacing: Voxel spacing in mm
        
    Returns:
        Surface Dice score
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Convert to binary
    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)
    
    # Handle empty arrays
    if np.sum(y_pred) == 0 and np.sum(y_true) == 0:
        return 1.0
    elif np.sum(y_pred) == 0 or np.sum(y_true) == 0:
        return 0.0
    
    from scipy import ndimage
    
    # Calculate distance transforms
    if spacing is not None:
        dt_true = distance_transform_edt(1 - y_true, sampling=spacing)
        dt_pred = distance_transform_edt(1 - y_pred, sampling=spacing)
    else:
        dt_true = distance_transform_edt(1 - y_true)
        dt_pred = distance_transform_edt(1 - y_pred)
    
    # Get surface points
    pred_surface = ((y_pred - ndimage.binary_erosion(y_pred)) > 0)
    true_surface = ((y_true - ndimage.binary_erosion(y_true)) > 0)
    
    # Count surface points within tolerance
    pred_surface_pts = np.sum(pred_surface)
    true_surface_pts = np.sum(true_surface)
    
    pred_true_within_tol = np.sum(pred_surface * (dt_true <= tolerance))
    true_pred_within_tol = np.sum(true_surface * (dt_pred <= tolerance))
    
    # Calculate surface Dice
    surface_dice = (pred_true_within_tol + true_pred_within_tol) / (pred_surface_pts + true_surface_pts)
    
    return float(surface_dice)


def compute_metrics(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    metrics: List[str] = ["dice", "iou", "precision", "recall"],
    spacing: Optional[Tuple[float, ...]] = None
) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics at once.
    
    Args:
        y_pred: Prediction array
        y_true: Ground truth array
        metrics: List of metrics to compute
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    for metric in metrics:
        if metric == "dice":
            results["dice"] = dice_score(y_pred, y_true)
        elif metric == "iou" or metric == "jaccard":
            results["iou"] = iou_score(y_pred, y_true)
        elif metric == "precision":
            results["precision"] = precision_score(y_pred, y_true)
        elif metric == "recall" or metric == "sensitivity":
            results["recall"] = recall_score(y_pred, y_true)
        elif metric == "specificity":
            results["specificity"] = specificity_score(y_pred, y_true)
        elif metric == "f1":
            results["f1"] = f1_score(y_pred, y_true)
        elif metric == "hausdorff":
            results["hausdorff"] = hausdorff_distance(y_pred, y_true, spacing=spacing)
        elif metric == "surface_dice":
            results["surface_dice"] = surface_dice(y_pred, y_true, spacing=spacing)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Calculate volume statistics
    if "volume" in metrics or "volume_error" in metrics:
        # Convert to binary
        if isinstance(y_pred, torch.Tensor):
            y_pred_bin = (y_pred > 0.5).detach().cpu().numpy()
        else:
            y_pred_bin = (y_pred > 0.5).astype(np.float32)
            
        if isinstance(y_true, torch.Tensor):
            y_true_bin = (y_true > 0.5).detach().cpu().numpy()
        else:
            y_true_bin = (y_true > 0.5).astype(np.float32)
        
        # Calculate volumes
        pred_vol = np.sum(y_pred_bin)
        true_vol = np.sum(y_true_bin)
        
        # Apply spacing if provided
        if spacing is not None:
            voxel_vol = np.prod(spacing)
            pred_vol *= voxel_vol
            true_vol *= voxel_vol
        
        # Store volumes and error metrics
        results["volume"] = float(true_vol)
        results["pred_volume"] = float(pred_vol)
        results["volume_error"] = float(abs(pred_vol - true_vol))
        
        if true_vol > 0:
            results["volume_error_ratio"] = float(abs(pred_vol - true_vol) / true_vol)
    
    return results