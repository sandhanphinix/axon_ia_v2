# Post-Training Analysis Script Usage Guide

## Overview
The refactored `run_post_training_analysis.py` script now uses YAML configuration files instead of command-line arguments, making it more flexible and reproducible.

## Quick Start

### 1. Update Configuration File
Edit `configs/analysis/example_analysis_config.yaml` with your specific paths:

```yaml
training:
  config_path: "configs/training/swinunetr_config.yaml"
  checkpoint_path: "C:/development/data/axon_ia/outputs/swinunetr/checkpoints/model_015.pth"  # Your best model

data:
  data_dir: "C:/development/data/axon_ia/processed"
  splits: ["val"]  # Add "test" if you have test data
```

### 2. Run Analysis
```bash
# Basic usage with default config
python scripts/run_post_training_analysis.py

# Use specific config file
python scripts/run_post_training_analysis.py --config configs/analysis/example_analysis_config.yaml

# Dry run (see what would be executed)
python scripts/run_post_training_analysis.py --dry-run

# Override specific values
python scripts/run_post_training_analysis.py \
    --override training.checkpoint_path=/path/to/different/model.pth \
    --override steps.visualization.num_samples=20
```

## Configuration File Structure

### Required Sections

#### Training Configuration
```yaml
training:
  config_path: "path/to/training/config.yaml"     # Your training config
  checkpoint_path: "path/to/model/checkpoint.pth"  # Best model checkpoint
```

#### Data Configuration
```yaml
data:
  data_dir: "path/to/processed/data"
  splits: ["val", "test"]  # Splits to evaluate
```

#### Output Configuration
```yaml
output:
  base_dir: "./analysis_results"
  create_timestamp_folder: true  # Creates unique folder per run
```

### Analysis Steps

#### Evaluation
```yaml
steps:
  evaluation:
    enabled: true
    skip_if_exists: false    # Skip if metrics.json already exists
    batch_size: 1
    save_predictions: true   # Save prediction NIfTI files
    generate_report: true    # Generate HTML/PDF reports
    metrics: ["dice", "iou", "hausdorff", "precision", "recall"]
```

#### Visualization
```yaml
  visualization:
    enabled: true
    skip_if_exists: false
    num_samples: 10          # Number of cases to visualize
    modality: "flair"        # Background modality for overlays
    slice_selection: "center" # How to select slices (center/max_lesion)
```

#### Analysis & Recommendations
```yaml
  analysis:
    enabled: true
    generate_recommendations: true
    performance_thresholds:
      excellent_dice: 0.8    # Thresholds for performance categories
      good_dice: 0.7
      fair_dice: 0.5
      min_precision: 0.7
      min_recall: 0.7
```

#### Reporting
```yaml
  reporting:
    enabled: true
    formats: ["markdown", "json"]  # Report formats to generate
```

## Command Line Options

- `--config`: Path to analysis config file (default: `configs/analysis/post_training_analysis_config.yaml`)
- `--override`: Override config values (format: `key.subkey=value`)
- `--dry-run`: Show what would be executed without running

## Override Examples

```bash
# Change checkpoint path
--override training.checkpoint_path=/new/path/model.pth

# Change number of visualization samples
--override steps.visualization.num_samples=20

# Disable visualization step
--override steps.visualization.enabled=false

# Change performance thresholds
--override steps.analysis.performance_thresholds.excellent_dice=0.85

# Add test split
--override data.splits=[\"val\",\"test\"]
```

## Output Structure

```
analysis_results/
├── analysis_20250714_143022/          # Timestamped folder
│   ├── evaluation_val/                # Evaluation results
│   │   ├── metrics.json              # Detailed metrics
│   │   └── predictions/              # Prediction NIfTI files
│   ├── visualizations_val/           # Visualization outputs
│   │   ├── sample_001_overlay.png
│   │   ├── sample_001_error_analysis.png
│   │   └── metrics_summary.png
│   ├── analysis_report.md            # Main analysis report
│   ├── analysis_report.json          # Machine-readable report
│   ├── recommendations.json          # Specific recommendations
│   └── full_config.yaml             # Complete config for reproducibility
```

## Benefits of YAML Configuration

1. **Reproducibility**: Complete configuration saved with results
2. **Flexibility**: Easy to create different analysis profiles
3. **Version Control**: Config files can be tracked in git
4. **Batch Processing**: Easy to run multiple analyses with different settings
5. **Documentation**: Self-documenting with comments in YAML

## Example Workflows

### Quick Validation Check
```yaml
# configs/analysis/quick_check.yaml
steps:
  evaluation:
    enabled: true
    metrics: ["dice", "iou"]
  visualization:
    enabled: true
    num_samples: 3
  analysis:
    enabled: true
  reporting:
    enabled: true
    formats: ["markdown"]
```

### Comprehensive Analysis
```yaml
# configs/analysis/comprehensive.yaml
data:
  splits: ["val", "test"]
steps:
  evaluation:
    enabled: true
    metrics: ["dice", "iou", "hausdorff", "precision", "recall", "specificity"]
  visualization:
    enabled: true
    num_samples: 20
  analysis:
    enabled: true
  reporting:
    enabled: true
    formats: ["markdown", "json"]
```

### Visualization Only
```yaml
# configs/analysis/vis_only.yaml
steps:
  evaluation:
    enabled: false
  visualization:
    enabled: true
    skip_if_exists: false
    num_samples: 15
  analysis:
    enabled: true  # Will use existing metrics
  reporting:
    enabled: true
```

## Troubleshooting

1. **Config validation errors**: Check that all required paths exist and are correct
2. **Permission errors**: Ensure write access to output directory
3. **Memory issues**: Reduce `num_samples` or `batch_size`
4. **Missing dependencies**: Ensure all required packages are installed

## Migration from CLI Version

Old CLI command:
```bash
python scripts/run_post_training_analysis.py \
    --config configs/training/swinunetr_config.yaml \
    --checkpoint /path/to/model.pth \
    --splits val test \
    --num-vis-samples 10 \
    --output-dir ./results
```

New YAML-based approach:
1. Create config file with these settings
2. Run: `python scripts/run_post_training_analysis.py --config your_config.yaml`

This provides better organization, reproducibility, and easier parameter management.
