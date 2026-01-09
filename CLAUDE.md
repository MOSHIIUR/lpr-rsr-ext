# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Single-Image Super-Resolution (SISR)** research implementation specifically designed for **License Plate Recognition (LPR)**. It combines deep learning with attention mechanisms and sub-pixel convolution to enhance low-resolution license plate images for improved OCR accuracy.

**Published Research**: "Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers" in *Computers & Graphics* (2023).

## Repository Structure

- **Proposed/**: Main implementation (latest version, use this for development)
- **Sibgrapi2022/**: Conference paper version (legacy)
- **Mehri/**: Alternative implementation (legacy)
- **saved_models/**: Pre-trained OCR models (RodoSol-SR for Brazilian plates, PKU-SR for Chinese plates)

## Setup and Installation

### 1. Create Conda Environment

```bash
# Create and activate conda environment from environment.yml
conda env create -f environment.yml
conda activate lpr-rsr
```

**What's included:**
- Python 3.9
- PyTorch + torchvision with CUDA 11.8 support
- TensorFlow + Keras (for pre-trained OCR models)
- OpenCV, scikit-image, albumentations (image processing)
- pandas, numpy, matplotlib (data handling)
- python-Levenshtein (OCR accuracy metrics)

**Alternative (pip-only installation):**
```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained OCR Models

Download from: https://github.com/Valfride/lpr-rsr-ext/releases/download/OCR_pre-trained_models/saved_models.zip

```bash
# Download and extract OCR models
wget https://github.com/Valfride/lpr-rsr-ext/releases/download/OCR_pre-trained_models/saved_models.zip
unzip saved_models.zip
```

Expected structure:
```
saved_models/
├── RodoSol-SR/
│   ├── model.json
│   └── parameters.json
└── PKU-SR/
    ├── model.json
    └── parameters.json
```

### 3. Verify GPU Setup (Optional but Recommended)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Commands

### Dataset Preparation

```bash
# Prepare custom ICPR dataset (creates dataset.txt and metadata files)
python prepare_dataset.py \
    --dataset-root /path/to/ICPR/dataset/train/Scenario-A/Brazilian \
    --output-dir ./dataset \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### Training

```bash
cd Proposed

# Train from scratch (mode 0)
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 16 -m 0

# Resume training from checkpoint (mode 1)
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 16 -m 1 --model ./checkpoints/bestmodel.pt
```

**Training outputs:**
- `bestmodel.pt`: Best model based on validation loss (use for testing)
- `backup.pt`: Latest checkpoint

### Testing

```bash
cd Proposed

python testing.py -t ../dataset/dataset.txt -s ./results -b 16 --model ./checkpoints/bestmodel.pt
```

**Testing outputs:**
- `eval.csv`: PSNR, SSIM, OCR accuracy metrics
- `SR_*.jpg/png`: Super-resolved images
- `HR_*.jpg/png`: Ground truth high-res images
- `LR_*.jpg/png`: Low-resolution input images

### Weights & Biases Integration

Track training metrics, sample images, and test results with W&B:

```bash
# Install wandb (optional)
pip install wandb
wandb login  # First time only

# Training with W&B
cd Proposed
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 16 -m 0 \
    --wandb-project my-lpr-project \
    --wandb-run-name experiment-001 \
    --wandb-log-interval 5

# Testing (automatically resumes the training run)
python testing.py \
    -t ../dataset/dataset.txt \
    -s ./results \
    -b 16 \
    --model ./checkpoints/bestmodel.pt

# Disable W&B
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 16 -m 0 --disable-wandb
```

**Tracked Metrics:**
- **Per-batch/step**: train/val batch losses (total, MSE, features), running averages
- **Sample images**: LR/HR/SR comparison every N epochs
- **Test**: PSNR, SSIM, OCR accuracy, detailed results table

**W&B Arguments:**
- `--wandb-project`: Project name (default: "lpr-super-resolution")
- `--wandb-run-name`: Custom run name (auto-generated if omitted)
- `--wandb-entity`: Team/organization name (optional)
- `--wandb-log-interval`: Image logging frequency in epochs (default: 5)
- `--disable-wandb`: Disable W&B logging
- `--wandb-run-id`: Resume specific run (testing only)

**Note:** wandb is completely optional. Training and testing work normally without it installed or when disabled with `--disable-wandb`.

## Critical Configuration Points

### 1. Image Dimensions (Proposed/__dataset__.py:22-23)

Must match your dataset's aspect ratio:

```python
# For RodoSol-SR (Brazilian plates, aspect_ratio=2.0)
IMG_LR = (40, 20)
IMG_HR = (160, 80)

# For PKU-SR (Chinese plates, aspect_ratio=3.0)
IMG_LR = (48, 16)
IMG_HR = (192, 64)
```

Also update `self.aspect_ratio` in the `customDataset` class constructor.

### 2. OCR Model Path

Update in both files:
- **Proposed/training.py:339** (search for `path_ocr = Path(`)
- **Proposed/testing.py:22** (search for `path_ocr = Path(`)

```python
# For Brazilian plates
path_ocr = Path('./saved_models/RodoSol-SR')

# For Chinese plates
path_ocr = Path('./saved_models/PKU-SR')
```

### 3. Dataset Format

**dataset.txt format** (semicolon-separated):
```
/path/to/HR_image1.jpg;/path/to/LR_image1.jpg;training
/path/to/HR_image2.jpg;/path/to/LR_image2.jpg;validation
/path/to/HR_image3.jpg;/path/to/LR_image3.jpg;testing
```

**Metadata files**: Each HR image must have a corresponding `.txt` file with the same base name:
```
Type: car
Plate: ABC1234
Layout: ABC-1234
Points: x1,y1 x2,y2 x3,y3 x4,y4
```

## Architecture Overview

### Network Architecture (Proposed/network.py)

The super-resolution network is organized as follows:

```
Input (LR Image)
    ↓
[AutoEncoder] - Depthwise-separable convolutions + PixelUnshuffle for feature extraction
    ↓
[RDN (32 blocks)] - Residual Dense Network for deep feature learning
    ↓
[ResidualModule] - Multi-layer residual blocks with concatenation
    ↓
[FeatureModule] - TFAM (Transformer-Inspired Attention) + feature fusion
    ↓
[RDN (18 blocks)] - Additional feature refinement
    ↓
[PixelShuffle 4x] - Sub-pixel convolution upsampling (2x → 2x)
    ↓
[Output] - Residual connection: SR features + upscaled input
    ↓
Output (SR Image)
```

**Key Components:**
- **TFAM (Transformer-Inspired Attention Module)**: Combines DPCA (Dual Path Channel Attention), POS (Position-aware sub-sampling), and CA (Channel-aware attention)
- **RDB (Residual Dense Block)**: Local dense connections with global residual learning
- **ARC (Adaptive Residual Block)**: Depthwise convolutions with attention
- **AutoEncoder**: Uses PixelUnshuffle/PixelShuffle for multi-scale processing

### Training Strategy

**Loss Function** (Proposed/training.py):
- Combined **MSE loss** + **OCR-based feature extraction loss** (L1 on intermediate features)
- Uses pre-trained Keras OCR models to extract features from HR, LR, and SR images
- OCR features guide the network to preserve text readability

**Optimization**:
- Adam optimizer (lr=0.0001)
- Learning rate scheduling: ReduceLROnPlateau (factor=0.8, patience=2)
- Early stopping: patience=20 epochs on validation loss
- Max epochs: 200

**Data Pipeline**:
- 70/15/15 train/val/test split
- Augmentation via albumentations (rotation, shift, blur, noise)
- Automatic brightness/contrast adjustment

### Dataset Loading (Proposed/__dataset__.py)

The `customDataset` class handles:
- Reading dataset.txt files with semicolon-separated paths
- Loading and preprocessing HR/LR image pairs
- Applying augmentation (training only)
- Padding images to maintain aspect ratio
- Automatic brightness/contrast normalization
- Reading metadata files for OCR ground truth

Split parsing pattern in `__getPaths__()`:
- Training: lines containing `'training'` in split field
- Validation: lines containing `'validation'` in split field
- Testing: lines containing `'testing'` in split field

## Important Notes for Development

### GPU Requirements
- Code uses CUDA by default (`model.cuda()`)
- Multi-GPU support available via device_list configuration
- For CPU-only development, modify `.cuda()` calls in network initialization

### Version Directory Comparison
- **Proposed/**: Latest implementation with TFAM, DPCA, and improved RDN
- **Sibgrapi2022/**: Conference version (earlier architecture)
- **Mehri/**: Alternative variation

Always work in **Proposed/** directory unless specifically researching architectural differences.

### File Organization Pattern
Each version directory (Proposed/, Sibgrapi2022/, Mehri/) contains:
- `training.py`: Main training loop
- `testing.py`: Evaluation script
- `network.py`: Model architecture
- `__dataset__.py`: PyTorch Dataset class
- `functions.py`: Utility functions (OCR loading, padding, etc.)
- `__parser__.py`: CLI argument parsing
- `__syslog__.py`: Error handling decorators

### Common Gotchas

1. **Hardcoded CUDA in network.py:241**: `nn.Conv2d(...).cuda()` creates layers on GPU directly
2. **OCR path mismatch**: Ensure paths in training.py and testing.py point to the same OCR model
3. **Aspect ratio consistency**: IMG_LR, IMG_HR, and self.aspect_ratio must all be consistent
4. **Dataset separator**: Uses semicolon `;` not comma for dataset.txt files
5. **Metadata requirement**: Testing will fail if HR images don't have corresponding .txt metadata files

### Citation

When referencing this work:
```bibtex
@article{nascimento2023super,
  title = {Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers},
  author = {V. {Nascimento} and R. {Laroca} and J. A. {Lambert} and W. R. {Schwartz} and D. {Menotti}},
  year = {2023},
  journal = {Computers \& Graphics},
  volume = {113},
  pages = {69-76},
  doi = {10.1016/j.cag.2023.05.005}
}
```
