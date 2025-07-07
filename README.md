# Axon IA v2

Axon IA is an advanced, modular framework for medical image segmentation, with a focus on brain MRI analysis and lesion segmentation. It provides state-of-the-art deep learning architectures, robust data processing, and comprehensive tools for training, inference, and evaluation.

---

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Supported Models](#supported-models)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features
- Advanced deep learning architectures: **UNETR**, **SwinUNETR**, **nnU-Net**, **SegResNet**
- Comprehensive 3D medical image data pipeline (preprocessing, augmentation, postprocessing)
- Specialized augmentation and preprocessing for MRI
- Efficient training: mixed precision, gradient accumulation, early stopping, model checkpointing
- Sliding-window inference with test-time augmentation
- Detailed evaluation metrics and automated report generation
- Visualization tools for data, predictions, and metrics
- Extensible configuration via YAML files

---

## System Architecture
Axon IA is organized into the following modules:
- **config**: Configuration parsing and management
- **data**: Dataset classes, transforms, augmentation, preprocessing
- **models**: Model architectures and factory
- **losses**: Loss functions for segmentation
- **training**: Trainer, callbacks, learning rate schedulers
- **inference**: Predictors, sliding window, postprocessing
- **evaluation**: Metrics, report generation, visualization
- **utils**: Logging, GPU management, NIfTI utilities, visualization

For a detailed diagram, see [`docs/architecture.md`](docs/architecture.md).

---

## Installation

### Requirements
- Python >= 3.8
- See `requirements.txt` for all dependencies

### Install
```bash
# Clone the repository
git clone https://github.com/yourusername/axon_ia_v2.git
cd axon_ia_v2

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Data Format
Axon IA expects data in a hierarchical directory structure, organized by dataset splits (train/val/test), with each case containing NIfTI files for each modality and the target mask. See [`docs/data_format.md`](docs/data_format.md) for details.

---

## Configuration
All experiment settings are controlled via YAML config files (see `configs/`).
- **Data**: modalities, target, preprocessing, augmentation
- **Model**: architecture, input/output channels, hyperparameters
- **Training**: epochs, batch size, optimizer, scheduler, callbacks
- **Inference**: sliding window, TTA, postprocessing

See [`axon_ia/config/default_config.yaml`](axon_ia/config/default_config.yaml) for a template.

---

## Usage

### Data Preparation
Preprocess raw data for training:
```bash
python scripts/prepare_data.py --input-dir <raw_data_dir> --output-dir <processed_data_dir> [--config <config.yaml>]
```
- Handles resampling, normalization, cropping, and metadata generation.

### Model Training
Train a segmentation model:
```bash
python scripts/train.py --config <config.yaml> [--data-dir <data_dir>] [--output-dir <output_dir>] [--model <architecture>]
```
- Supports early stopping, checkpointing, TensorBoard, and Weights & Biases logging.

### Inference
Run inference on new data:
```bash
python scripts/predict.py --config <config.yaml> --checkpoint <model.pth> --data-dir <data_dir> --output-dir <output_dir>
```
- Sliding window inference, TTA, and postprocessing are configurable.

### Evaluation
Evaluate a trained model:
```bash
python scripts/evaluate.py --config <config.yaml> --checkpoint <model.pth> --data-dir <data_dir> --output-dir <output_dir>
```
- Computes metrics (Dice, IoU, Hausdorff, etc.) and generates reports/visualizations.

---

## Supported Models
- **UNETR**
- **SwinUNETR**
- **nnU-Net**
- **SegResNet**

Model selection and parameters are set in the config file.

---

## Advanced Features
- **Custom Losses**: Dice, Focal, Combo, Boundary, and more
- **Callbacks**: Early stopping, model checkpoint, TensorBoard, WandB
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
- **GPU Utilities**: Device selection, memory optimization, benchmarking
- **Visualization**: Plot slices, overlays, metrics, and reports
- **Export**: Convert models to ONNX for deployment (`scripts/export_model.py`)

---

## Testing
Run all tests with:
```bash
pytest tests/
```

---

## Contributing
Contributions are welcome! Please open issues or pull requests. See the code style in `.flake8`, `black`, and `isort` configs.

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

---

## Contact
For questions or support, contact the Axon IA Development Team at info@axonia.org.