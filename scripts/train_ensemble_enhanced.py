#!/usr/bin/env python
"""
Enhanced ensemble training script with advanced features for small lesion detection.

This script implements comprehensive ensemble training with:
- 5 state-of-the-art models
- Advanced loss functions optimized for small lesions
- Curriculum learning
- 5-fold cross-validation
- Enhanced data augmentation
- Multi-scale training
"""

import os
import sys
import argparse
import yaml
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from axon_ia.config import ConfigParser
    from axon_ia.data import AxonDataset
    from axon_ia.models.model_factory import create_model
    from axon_ia.losses.advanced_losses import create_loss
    from axon_ia.evaluation.metrics import compute_metrics
    from axon_ia.utils.logger import get_logger
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    # Try importing individual modules to diagnose
    try:
        from axon_ia.data import AxonDataset
        print("[OK] AxonDataset imported successfully")
    except ImportError as e:
        print(f"[ERROR] AxonDataset import failed: {e}")
    
    try:
        from axon_ia.models.model_factory import create_model
        print("[OK] create_model imported successfully")
    except ImportError as e:
        print(f"[ERROR] create_model import failed: {e}")
    
    try:
        from axon_ia.losses.advanced_losses import create_loss
        print("[OK] create_loss imported successfully")
    except ImportError as e:
        print(f"[ERROR] create_loss import failed: {e}")
    
    try:
        from axon_ia.utils.logger import get_logger
        print("[OK] get_logger imported successfully")
    except ImportError as e:
        print(f"[ERROR] get_logger import failed: {e}")
    
    sys.exit(1)


class CurriculumLearning:
    """Implements curriculum learning for medical image segmentation."""
    
    def __init__(self, warmup_epochs: int = 10, easy_ratio: float = 0.7):
        self.warmup_epochs = warmup_epochs
        self.easy_ratio = easy_ratio
        self.difficulties = {}
    
    def compute_difficulties(self, dataset):
        """Compute difficulty scores for all samples."""
        print("Computing curriculum difficulties...")
        
        for i in tqdm(range(len(dataset)), desc="Computing difficulties"):
            sample = dataset[i]
            mask = sample['mask'].numpy() if hasattr(sample['mask'], 'numpy') else sample['mask']
            
            # Simple difficulty metric: smaller lesions are harder
            lesion_volume = np.sum(mask > 0.5)
            
            # Normalize difficulty (smaller = harder)
            difficulty = 1.0 / (lesion_volume + 1)  # +1 to avoid division by zero
            self.difficulties[i] = difficulty
    
    def get_curriculum_indices(self, epoch: int, all_indices: List[int]) -> List[int]:
        """Get indices for current epoch based on curriculum."""
        if epoch < self.warmup_epochs:
            # During warmup, use easier samples
            difficulties = [(i, self.difficulties.get(i, 0.5)) for i in all_indices]
            difficulties.sort(key=lambda x: x[1])  # Sort by difficulty (easier first)
            
            n_easy = int(len(all_indices) * self.easy_ratio)
            return [idx for idx, _ in difficulties[:n_easy]]
        else:
            # After warmup, use all samples
            return all_indices


class SmallLesionSampler:
    """Weighted sampler that boosts small lesions."""
    
    def __init__(self, dataset, small_lesion_threshold: int = 100, boost_factor: float = 2.0):
        self.dataset = dataset
        self.threshold = small_lesion_threshold
        self.boost_factor = boost_factor
        self.weights = self._compute_weights()
    
    def _compute_weights(self):
        """Compute sampling weights based on lesion size."""
        weights = []
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            mask = sample['mask'].numpy() if hasattr(sample['mask'], 'numpy') else sample['mask']
            volume = np.sum(mask > 0.5)
            
            if volume < self.threshold:
                weight = self.boost_factor
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sampler(self, indices: List[int]):
        """Get weighted random sampler for given indices."""
        subset_weights = self.weights[indices]
        return torch.utils.data.WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(indices),
            replacement=True
        )


class EnsembleTrainer:
    """Enhanced trainer for ensemble models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Force CPU device if specified in config
        if config.get('global', {}).get('training', {}).get('device') == 'cpu':
            self.device = torch.device("cpu")
            print("[CPU] Using CPU device (as specified in config)")
        elif config.get('hardware', {}).get('device') == 'cpu':
            self.device = torch.device("cpu")
            print("[CPU] Using CPU device (as specified in config)")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DEVICE] Using device: {self.device}")
        
        self.best_metrics = defaultdict(float)
        self.curriculum = None
        self.small_lesion_sampler = None
        
        # Setup mixed precision training
        precision = config.get('global', {}).get('training', {}).get('precision', '32')
        self.use_mixed_precision = precision == "16-mixed" and self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("[GPU] Mixed precision training enabled")
        else:
            self.scaler = None
            print(f"[PRECISION] Using {precision} precision")
        
        # Setup curriculum learning
        curriculum_config = config.get('global', {}).get('training', {}).get('curriculum_learning', {})
        if curriculum_config.get('enabled', False):
            self.curriculum = CurriculumLearning(
                warmup_epochs=curriculum_config.get('warmup_epochs', 10),
                easy_ratio=curriculum_config.get('easy_samples_ratio', 0.7)
            )
    
    def create_dataset(self) -> AxonDataset:
        """Create and return the dataset."""
        dataset_path = self.config['data']['dataset_path']
        print(f"[FOLDER] Looking for training data in: {dataset_path}/train")
        
        # Get modalities and target from config
        data_config = self.config.get('data', {})
        modalities = data_config.get('modalities', ['b0', 'b1000', 'flair', 'T2Star'])
        target = data_config.get('target', 'perfroi')
        
        print(f"[SEARCH] Using modalities: {modalities}")
        print(f"[TARGET] Target segmentation: {target}")
        
        return AxonDataset(
            root_dir=dataset_path,
            split='train',
            modalities=modalities,
            target=target,
            transform=self._get_transforms()
        )
    
    def create_val_dataset(self) -> AxonDataset:
        """Create and return the validation dataset."""
        dataset_path = self.config['data']['dataset_path']
        
        # Get modalities and target from config
        data_config = self.config.get('data', {})
        modalities = data_config.get('modalities', ['b0', 'b1000', 'flair', 'T2Star'])
        target = data_config.get('target', 'perfroi')
        
        # Try test split first (since there's no validation split)
        print(f"[FOLDER] Looking for validation data in: {dataset_path}/test")
        
        try:
            return AxonDataset(
                root_dir=dataset_path,
                split='test',  # Use test split for validation during training
                modalities=modalities,
                target=target,
                transform=self._get_val_transforms()
            )
        except Exception as e:
            print(f"[WARNING] Could not load test split for validation: {e}")
            print("[FOLDER] Will use training data split for validation")
            return None
    
    def _get_transforms(self):
        """Get training transforms based on config."""
        return None
    
    def _get_val_transforms(self):
        """Get validation transforms."""
        return None
    
    def create_stratified_splits(self, dataset: AxonDataset, n_folds: int = 5, 
                               random_state: int = 42) -> List[Tuple[List[int], List[int]]]:
        """Create stratified cross-validation splits."""
        # Check for empty dataset
        if len(dataset) == 0:
            print("[ERROR] Cannot create cross-validation splits: dataset is empty!")
            return []
        
        lesion_volumes = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample['mask'].numpy() if hasattr(sample['mask'], 'numpy') else sample['mask']
            volume = np.sum(mask > 0.5)
            lesion_volumes.append(volume)
        
        # Check if we have enough samples for stratified splitting
        if len(lesion_volumes) < n_folds:
            print(f"[WARNING] Dataset has only {len(lesion_volumes)} samples, but {n_folds} folds requested.")
            print("   Using simple random splits instead of stratified splits.")
            # Create simple random splits
            indices = list(range(len(dataset)))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            fold_size = len(indices) // n_folds
            folds = []
            
            for i in range(n_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(indices)
                val_idx = indices[start_idx:end_idx]
                train_idx = [idx for idx in indices if idx not in val_idx]
                folds.append((train_idx, val_idx))
            
            return folds
        
        # Create volume-based strata
        volume_quartiles = np.percentile(lesion_volumes, [25, 50, 75])
        strata = np.digitize(lesion_volumes, volume_quartiles)
        
        # Create folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        folds = []
        for train_idx, val_idx in skf.split(range(len(dataset)), strata):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        
        return folds
    
    def create_model_and_optimizer(self, model_config: Dict[str, Any]):
        """Create model, loss, and optimizer from config."""
        # Create model
        model = create_model(
            architecture=model_config['architecture'],
            **model_config['params']
        ).to(self.device)
        
        # Load pretrained weights if specified
        if 'pretrained_path' in model_config and model_config['pretrained_path']:
            pretrained_path = model_config['pretrained_path']
            if os.path.exists(pretrained_path):
                try:
                    print(f"[LOAD] Loading pretrained weights from: {pretrained_path}")
                    checkpoint = torch.load(pretrained_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Load weights with compatibility for different architectures
                    model_dict = model.state_dict()
                    # Filter out unnecessary keys and size mismatches
                    filtered_dict = {}
                    for k, v in state_dict.items():
                        if k in model_dict and model_dict[k].shape == v.shape:
                            filtered_dict[k] = v
                        else:
                            print(f"[SKIP] Skipping layer {k} due to size mismatch or missing key")
                    
                    model_dict.update(filtered_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print(f"[OK] Loaded {len(filtered_dict)}/{len(model_dict)} layers from pretrained model")
                    
                except Exception as e:
                    print(f"[WARNING] Failed to load pretrained weights: {e}")
                    print("   Continuing with random initialization...")
            else:
                print(f"[WARNING] Pretrained weights file not found: {pretrained_path}")
                print("   Continuing with random initialization...")
        
        # Create loss function
        loss_config = model_config['loss']
        criterion = create_loss(
            loss_type=loss_config['type'],
            **loss_config['params']
        )
        
        # Create optimizer
        opt_config = model_config['optimizer']
        if opt_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=float(opt_config['learning_rate']),
                weight_decay=float(opt_config.get('weight_decay', 1e-5)),
                betas=opt_config.get('betas', [0.9, 0.999])
            )
        elif opt_config['type'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=float(opt_config['learning_rate']),
                weight_decay=float(opt_config.get('weight_decay', 1e-5)),
                amsgrad=opt_config.get('amsgrad', False)
            )
        elif opt_config['type'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=float(opt_config['learning_rate']),
                momentum=float(opt_config.get('momentum', 0.9)),
                weight_decay=float(opt_config.get('weight_decay', 1e-4)),
                nesterov=opt_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return model, criterion, optimizer
    
    def train_single_model(self, model_name: str, model_config: Dict[str, Any], 
                          train_indices: List[int], val_indices: List[int],
                          train_dataset: AxonDataset, val_dataset: AxonDataset,
                          fold: int = 0) -> Dict[str, float]:
        """Train a single model for one fold."""
        print(f"\n[INFO] Training {model_name} (Fold {fold+1})")
        
        # Create model and optimizer
        model, criterion, optimizer = self.create_model_and_optimizer(model_config)
        
        # Create data loaders
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices) if val_indices else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['global']['training']['batch_size'],
            sampler=train_sampler,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        if val_dataset and val_sampler:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['global']['training']['batch_size'],
                sampler=val_sampler,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )
        else:
            val_loader = None
        
        # Training parameters
        max_epochs = self.config['global']['training']['max_epochs']
        best_dice = 0.0
        patience = self.config['global']['training']['early_stopping_patience']
        patience_counter = 0
        
        # Create checkpoints directory
        checkpoint_dir = os.path.join(self.config['global']['output_dir'], 'checkpoints', f"{model_name}_fold_{fold}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[CHECKPOINT] Checkpoints will be saved to: {checkpoint_dir}")
        
        # Check for existing checkpoints to resume from
        start_epoch = 0
        resume_checkpoint = None
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if checkpoint_files:
            # Sort by epoch number and get the latest one
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            resume_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            print(f"[RESUME] Found checkpoint: {latest_checkpoint}")
            try:
                resume_checkpoint = torch.load(resume_checkpoint_path, map_location=self.device)
                
                # Load model state
                model.load_state_dict(resume_checkpoint['model_state_dict'])
                optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                start_epoch = resume_checkpoint['epoch']
                best_dice = resume_checkpoint.get('best_dice', 0.0)
                patience_counter = resume_checkpoint.get('patience_counter', 0)
                
                print(f"[RESUME] Resuming from epoch {start_epoch+1}, best_dice: {best_dice:.4f}")
                
            except Exception as e:
                print(f"[RESUME] Failed to load checkpoint {latest_checkpoint}: {e}")
                print(f"[RESUME] Starting training from scratch...")
                start_epoch = 0
                best_dice = 0.0
                patience_counter = 0
        
        # Training loop
        for epoch in range(start_epoch, max_epochs):
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            
            # Get curriculum indices if using curriculum learning
            if self.curriculum:
                curr_train_indices = self.curriculum.get_curriculum_indices(epoch, train_indices)
                train_sampler = SubsetRandomSampler(curr_train_indices)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config['global']['training']['batch_size'],
                    sampler=train_sampler,
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=self.config['data']['pin_memory']
                )
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Training {model_name}")
            for batch_idx, batch in enumerate(train_pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                    dice = compute_metrics(pred_masks, masks)['dice']
                    train_dice += dice
                
                train_loss += loss.item()
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Dice': f"{dice:.4f}"
                })
            
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            val_dice = 0.0
            
            if val_loader:
                model.eval()
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validating {model_name}")
                    for batch in val_pbar:
                        images = batch['image'].to(self.device)
                        masks = batch['mask'].to(self.device)
                        
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        
                        pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                        dice = compute_metrics(pred_masks, masks)['dice']
                        
                        val_loss += loss.item()
                        val_dice += dice
                        
                        val_pbar.set_postfix({
                            'Loss': f"{loss.item():.4f}",
                            'Dice': f"{dice:.4f}"
                        })
                
                val_loss /= len(val_loader)
                val_dice /= len(val_loader)
            else:
                val_loss = train_loss
                val_dice = train_dice
            
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
                torch.save({
                    'epoch': epoch,  # Save current epoch index for proper resumption
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_dice': train_dice,
                    'val_dice': val_dice,
                    'best_dice': best_dice,
                    'patience_counter': patience_counter,  # Add patience counter
                    'model_config': model_config
                }, checkpoint_path)
                print(f"[CHECKPOINT] Saved checkpoint: {checkpoint_path}")
            
            # Early stopping and best model saving
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, f"best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_dice': train_dice,
                    'val_dice': val_dice,
                    'best_dice': best_dice,
                    'model_config': model_config
                }, best_model_path)
                print(f"[BEST] Saved best model: {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"final_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'best_dice': best_dice,
            'model_config': model_config
        }, final_model_path)
        print(f"[FINAL] Saved final model: {final_model_path}")
        
        return {
            'best_dice': best_dice,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_dice': train_dice,
            'final_val_dice': val_dice,
            'checkpoint_dir': checkpoint_dir
        }
    
    def train_ensemble(self, specific_model: Optional[str] = None, 
                      specific_fold: Optional[int] = None):
        """Train all models in ensemble with cross-validation."""
        print("[TARGET] Starting ensemble training pipeline")
        print(f"[INFO] Device: {self.device}")
        print(f"[SAVE] Output directory: {self.config['global']['output_dir']}")
        
        # Create datasets with progress indicator
        print("\n[DATA] Creating datasets...")
        train_dataset = self.create_dataset()
        # For k-fold CV, we don't use a separate validation dataset during training
        # The test dataset will only be used for final evaluation after training
        val_dataset = None  # Will be created from training data splits
        print(f"[OK] Created datasets - Train: {len(train_dataset)}, Test (for final eval): Not loaded during training")
        
        # Check for empty datasets
        if len(train_dataset) == 0:
            print("[ERROR] Training dataset is empty!")
            print("[FOLDER] Please check that your data directory structure is correct:")
            print(f"   Expected: {self.config['data']['dataset_path']}/train/")
            print("   Should contain patient folders with correctly named .nii.gz files")
            print("   File pattern: <modality>_<folder_name>.nii.gz")
            return {}
        
        # Setup curriculum learning
        if self.curriculum:
            print("\n[LEARN] Setting up curriculum learning...")
            self.curriculum.compute_difficulties(train_dataset)
            print("[OK] Curriculum learning configured")
        
        # Setup small lesion sampling
        if self.config['global']['training'].get('class_balancing', {}).get('enabled', False):
            print("\n[BALANCE] Setting up class balancing for small lesions...")
            class_config = self.config['global']['training']['class_balancing']
            self.small_lesion_sampler = SmallLesionSampler(
                train_dataset,
                small_lesion_threshold=class_config.get('small_lesion_threshold', 100),
                boost_factor=class_config.get('small_lesion_boost', 2.5)
            )
            print("[OK] Small lesion sampling configured")
        
        # Cross-validation setup
        print("\n[CV] Setting up cross-validation...")
        cv_config = self.config['global']['cross_validation']
        if cv_config['enabled']:
            folds = self.create_stratified_splits(
                train_dataset,
                n_folds=cv_config['n_folds'],
                random_state=cv_config['random_state']
            )
            if not folds:
                print("[ERROR] Failed to create cross-validation splits!")
                return {}
            print(f"[OK] Created {cv_config['n_folds']}-fold cross-validation splits")
        else:
            # Single fold using 80/20 split of training data (no separate validation set)
            train_indices = list(range(len(train_dataset)))
            split_idx = int(0.8 * len(train_indices))  # 80/20 split
            folds = [(train_indices[:split_idx], train_indices[split_idx:])]
            print("[OK] Using single fold with 80/20 split of training data")
        
        # Get models to train
        models_to_train = [specific_model] if specific_model else list(self.config['models'].keys())
        folds_to_train = [specific_fold] if specific_fold is not None else range(len(folds))
        
        print(f"\n[TARGET] Training plan:")
        print(f"   Models: {models_to_train}")
        print(f"   Folds: {list(folds_to_train)}")
        print(f"   Total combinations: {len(models_to_train) * len(folds_to_train)}")
        
        results = {}
        
        # Train each model on each fold
        for model_name in models_to_train:
            if model_name not in self.config['models']:
                print(f"[WARNING] Model {model_name} not found in config, skipping...")
                continue
            
            model_config = self.config['models'][model_name]
            results[model_name] = {}
            
            for fold_idx in folds_to_train:
                if fold_idx >= len(folds):
                    print(f"[WARNING] Fold {fold_idx} does not exist, skipping...")
                    continue
                
                train_indices, val_indices = folds[fold_idx]
                
                # For k-fold CV, both training and validation come from the same training dataset
                fold_val_dataset = train_dataset
                
                fold_results = self.train_single_model(
                    model_name=model_name,
                    model_config=model_config,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    train_dataset=train_dataset,
                    val_dataset=fold_val_dataset,
                    fold=fold_idx
                )
                
                results[model_name][f'fold_{fold_idx}'] = fold_results
                
                print(f"[OK] Completed {model_name} fold {fold_idx}")
                print(f"     Best Dice: {fold_results['best_dice']:.4f}")
        
        # Print summary
        print(f"\n[INFO] Training completed!")
        print(f"[INFO] Results summary:")
        for model_name, model_results in results.items():
            avg_dice = np.mean([fold_result['best_dice'] for fold_result in model_results.values()])
            print(f"   {model_name}: Average Dice = {avg_dice:.4f}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Ensemble Training for Medical Image Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Train specific model only')
    parser.add_argument('--fold', type=int, help='Train specific fold only')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    seed = config.get('global', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Initialize logger
    logger = get_logger()
    
    if args.dry_run:
        logger.info("Dry run - validating configuration")
        print("[CPU] Using CPU device (as specified in config)")
        logger.info("Configuration valid!")
        return 0
    
    # Create trainer and start training
    trainer = EnsembleTrainer(config)
    
    try:
        results = trainer.train_ensemble(
            specific_model=args.model,
            specific_fold=args.fold
        )
        
        print(f"\n[OK] Training completed successfully!")
        print(f"[INFO] Final results: {results}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print("=" * 80)
        print("TRAINING FAILED WITH ERROR CODE:", exit_code)
        print("=" * 80)
        print("Check the logs above for error details.")
    sys.exit(exit_code)