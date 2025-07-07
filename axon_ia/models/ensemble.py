"""
Model ensemble implementation for medical image segmentation.

This module provides functionality for combining multiple segmentation
models into an ensemble for improved performance.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Callable


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple segmentation models.
    
    This class combines predictions from multiple models
    using various ensemble methods to improve segmentation accuracy.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "mean",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of models to ensemble
            ensemble_method: Method for combining predictions ('mean', 'vote', 'max')
            weights: Optional weights for each model in weighted average
        """
        super(ModelEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        # Normalize weights if provided
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Convert to tensor and normalize
            weights = torch.tensor(weights)
            self.weights = weights / weights.sum()
        else:
            self.weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        # Get individual model predictions
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                
                # Handle case where model returns tuple/list (e.g., deep supervision)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]  # Take main output
                
                predictions.append(pred)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == "mean":
            # Weighted or simple average
            if self.weights is not None:
                # Apply weights
                weighted_sum = torch.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    weighted_sum += pred * self.weights[i].to(pred.device)
                ensemble_pred = weighted_sum
            else:
                # Simple average
                ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        elif self.ensemble_method == "vote":
            # Apply sigmoid/softmax first
            if predictions[0].size(1) == 1:
                # Binary case
                binary_preds = [torch.sigmoid(pred) > 0.5 for pred in predictions]
                # Count votes
                votes = torch.stack(binary_preds).sum(dim=0).float()
                # Majority vote
                ensemble_pred = (votes > len(self.models) / 2).float()
            else:
                # Multi-class case
                class_preds = [torch.argmax(pred, dim=1, keepdim=True) for pred in predictions]
                # Count votes for each class
                ensemble_pred = torch.zeros_like(predictions[0])
                for pred in class_preds:
                    for c in range(predictions[0].size(1)):
                        ensemble_pred[:, c:c+1] += (pred == c).float()
                # Select class with most votes
                ensemble_pred = torch.argmax(ensemble_pred, dim=1, keepdim=True)
        
        elif self.ensemble_method == "max":
            # Take maximum confidence
            if predictions[0].size(1) == 1:
                # Binary case
                probs = [torch.sigmoid(pred) for pred in predictions]
                ensemble_pred = torch.stack(probs).max(dim=0)[0]
            else:
                # Multi-class case
                probs = [torch.softmax(pred, dim=1) for pred in predictions]
                ensemble_pred = torch.stack(probs).max(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred
    
    def train(self, mode: bool = True) -> nn.Module:
        """
        Set training mode for all models.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            Self
        """
        for model in self.models:
            model.train(mode)
        return self
    
    def eval(self) -> nn.Module:
        """
        Set evaluation mode for all models.
        
        Returns:
            Self
        """
        for model in self.models:
            model.eval()
        return self