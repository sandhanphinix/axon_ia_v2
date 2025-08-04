"""
Advanced loss functions for ensemble training.

This module provides sophisticated loss functions designed to improve
small lesion detection and overall segmentation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalDiceLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss with enhanced small lesion detection.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_squared: bool = True,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_squared = dice_squared
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        if self.dice_squared:
            dice_loss = 1 - (2 * intersection + self.smooth) / (
                pred_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
            )
        else:
            dice_loss = 1 - (2 * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
            )
        
        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class ComboLossV2(nn.Module):
    """
    Advanced combination loss with boundary and surface components.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.6,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_squared: bool = True,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_squared = dice_squared
        self.smooth = smooth
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Boundary loss to improve edge prediction.
        """
        # Compute gradients to find boundaries
        pred_grad = torch.abs(torch.gradient(pred, dim=2)[0]) + \
                   torch.abs(torch.gradient(pred, dim=3)[0]) + \
                   torch.abs(torch.gradient(pred, dim=4)[0])
        
        target_grad = torch.abs(torch.gradient(target, dim=2)[0]) + \
                     torch.abs(torch.gradient(target, dim=3)[0]) + \
                     torch.abs(torch.gradient(target, dim=4)[0])
        
        return F.mse_loss(pred_grad, target_grad)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        if self.dice_squared:
            dice_loss = 1 - (2 * intersection + self.smooth) / (
                pred_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
            )
        else:
            dice_loss = 1 - (2 * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
            )
        
        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Boundary Loss
        boundary_loss = self.boundary_loss(pred, target)
        
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.boundary_weight * boundary_loss)


class TverskyFocalLoss(nn.Module):
    """
    Tversky loss combined with focal loss for handling class imbalance.
    """
    
    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 2.5,
        focal_weight: float = 0.5,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.alpha = tversky_alpha
        self.beta = tversky_beta
        self.gamma = focal_gamma
        self.focal_weight = focal_weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Tversky Loss
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1 - tversky
        
        # Focal component
        focal_tversky = tversky_loss ** (1 / self.gamma)
        
        # Standard focal loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        return (1 - self.focal_weight) * focal_tversky + self.focal_weight * focal_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining multiple loss functions.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.4,
        focal_weight: float = 0.3,
        lovasz_weight: float = 0.2,
        boundary_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
    
    def lovasz_hinge(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Lovasz hinge loss for IoU optimization.
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Convert to +1/-1 labels
        target_flat = 2 * target_flat - 1
        
        # Compute signs
        signs = target_flat * pred_flat
        errors = 1 - signs
        
        # Sort errors in descending order
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_sorted = target_flat[perm]
        
        # Compute gradient
        gt_sorted = target_sorted.sum()
        intersection = gt_sorted - target_sorted.cumsum(0)
        union = gt_sorted + (1 - target_sorted).cumsum(0)
        jaccard = 1 - intersection / union
        
        # Lovasz extension
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
        return torch.dot(errors_sorted, jaccard)
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Boundary loss using gradient computation.
        """
        # Compute spatial gradients
        pred_grad_x = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        pred_grad_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        pred_grad_z = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        
        target_grad_x = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        target_grad_y = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        target_grad_z = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        loss_z = F.mse_loss(pred_grad_z, target_grad_z)
        
        return (loss_x + loss_y + loss_z) / 3.0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Lovasz Loss
        lovasz_loss = self.lovasz_hinge(pred, target)
        
        # Boundary Loss
        boundary_loss = self.boundary_loss(pred, target)
        
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.lovasz_weight * lovasz_loss + 
                self.boundary_weight * boundary_loss)


class SmallLesionFocalLoss(nn.Module):
    """
    Specialized loss for improving small lesion detection.
    """
    
    def __init__(
        self,
        small_lesion_threshold: int = 50,
        small_lesion_weight: float = 3.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.6,
        focal_weight: float = 0.4,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.small_lesion_threshold = small_lesion_threshold
        self.small_lesion_weight = small_lesion_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
    
    def get_lesion_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weights based on lesion size.
        """
        weights = torch.ones_like(target)
        
        # Label connected components to identify individual lesions
        from scipy import ndimage
        
        for batch_idx in range(target.shape[0]):
            target_np = target[batch_idx, 0].cpu().numpy()
            labeled, num_features = ndimage.label(target_np > 0.5)
            
            for lesion_id in range(1, num_features + 1):
                lesion_mask = labeled == lesion_id
                lesion_size = np.sum(lesion_mask)
                
                if lesion_size < self.small_lesion_threshold:
                    # Boost weight for small lesions
                    weights[batch_idx, 0][torch.from_numpy(lesion_mask).to(target.device)] = self.small_lesion_weight
        
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Get lesion-size-based weights
        weights = self.get_lesion_weights(target)
        
        # Weighted Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        weights_flat = weights.view(-1)
        
        weighted_intersection = (pred_flat * target_flat * weights_flat).sum()
        weighted_sum = (pred_flat * weights_flat).sum() + (target_flat * weights_flat).sum()
        dice_loss = 1 - (2 * weighted_intersection + self.smooth) / (weighted_sum + self.smooth)
        
        # Weighted Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss * weights
        focal_loss = focal_loss.mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class SmallLesionLoss(nn.Module):
    """
    Specialized loss for small lesion detection with adaptive weighting.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.2,
        focal_gamma: float = 3.0,
        focal_alpha: float = 0.3,
        small_lesion_boost: float = 2.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.small_lesion_boost = small_lesion_boost
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate lesion size for adaptive weighting
        lesion_size = target.sum().float()
        total_size = target.numel()
        size_ratio = lesion_size / total_size
        
        # Boost loss for small lesions
        size_boost = self.small_lesion_boost if size_ratio < 0.01 else 1.0
        
        # Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Focal Loss with adaptive alpha
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Boundary Loss for edge preservation
        boundary_loss = self._boundary_loss(pred, target)
        
        total_loss = (
            self.dice_weight * dice_loss + 
            self.focal_weight * focal_loss + 
            self.boundary_weight * boundary_loss
        ) * size_boost
        
        return total_loss
    
    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss using gradients."""
        # Sobel filters for 3D
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                              dtype=pred.dtype, device=pred.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                              dtype=pred.dtype, device=pred.device)
        
        # Compute gradients
        pred_grad_x = F.conv3d(pred, sobel_x.unsqueeze(0), padding=1)
        pred_grad_y = F.conv3d(pred, sobel_y.unsqueeze(0), padding=1)
        target_grad_x = F.conv3d(target, sobel_x.unsqueeze(0), padding=1)
        target_grad_y = F.conv3d(target, sobel_y.unsqueeze(0), padding=1)
        
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.mse_loss(pred_grad, target_grad)


class AdaptiveFocalDice(nn.Module):
    """
    Adaptive Focal Dice Loss with label smoothing and dynamic alpha.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.6,
        focal_weight: float = 0.4,
        adaptive_alpha: bool = True,
        focal_gamma: float = 2.5,
        label_smoothing: float = 0.1,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.adaptive_alpha = adaptive_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            target_smooth = target * (1 - self.label_smoothing) + \
                           self.label_smoothing * 0.5
        else:
            target_smooth = target
        
        # Dice Loss
        pred_flat = pred.view(-1)
        target_flat = target_smooth.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Adaptive Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.adaptive_alpha:
            # Compute class frequencies
            pos_freq = target.mean()
            neg_freq = 1 - pos_freq
            alpha_t = neg_freq * target + pos_freq * (1 - target)
        else:
            alpha_t = 0.25 * target + 0.75 * (1 - target)
        
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class TverskyFocal(nn.Module):
    """
    Combination of Tversky Loss and Focal Loss for handling class imbalance.
    """
    
    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 2.0,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.alpha = tversky_alpha
        self.beta = tversky_beta
        self.gamma = focal_gamma
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # Tversky Loss
        tp = (pred_sigmoid * target).sum()
        fp = (pred_sigmoid * (1 - target)).sum()
        fn = ((1 - pred_sigmoid) * target).sum()
        
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tversky_loss = 1 - tversky_index
        
        # Focal component
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        return (tversky_loss * focal_weight.mean()).mean()


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss as described in the literature.
    """
    
    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        lambda_u: float = 1.0
    ):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.lambda_u = lambda_u
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # Symmetric Unified Focal loss
        loss_pos = -self.weight * target * torch.pow(1 - pred_sigmoid, self.gamma) * \
                   torch.log(pred_sigmoid + 1e-8)
        loss_neg = -(1 - self.weight) * (1 - target) * torch.pow(pred_sigmoid, self.gamma) * \
                   torch.log(1 - pred_sigmoid + 1e-8)
        
        loss = loss_pos + loss_neg
        
        # Asymmetric weighting
        asymmetric_weight = torch.pow(1 - pred_sigmoid, self.delta) * target + \
                           torch.pow(pred_sigmoid, self.delta) * (1 - target)
        
        return (self.lambda_u * loss * asymmetric_weight).mean()


# Loss factory for easy creation
def create_loss(loss_type: str, **kwargs):
    """
    Factory function to create loss functions.
    """
    loss_functions = {
        "focal_dice": FocalDiceLoss,
        "combo_loss_v2": ComboLossV2,
        "small_lesion_loss": SmallLesionLoss,
        "adaptive_focal_dice": AdaptiveFocalDice,
        "tversky_focal": TverskyFocal,
        "unified_focal_loss": UnifiedFocalLoss,
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)
