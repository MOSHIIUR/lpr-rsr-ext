# Training Guide - LPR Super-Resolution

Complete guide for training the License Plate Recognition Super-Resolution model.

## Table of Contents
1. [Training Overview](#training-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Modes](#training-modes)
4. [Hyperparameters](#hyperparameters)
5. [Monitoring Training](#monitoring-training)
6. [Checkpointing](#checkpointing)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

---

## Training Overview

The model performs 4x super-resolution on license plate images (32×16 → 128×64) using:
- **RDN (Residual Dense Network)** with 32 and 18 blocks for feature extraction
- **TFAM (Transformer-Inspired Attention Module)** for spatial attention
- **AutoEncoder** for initial feature compression
- **OCR-based loss** using pre-trained Keras models (RodoSol-SR)

**Loss Function:**
```
Total Loss = MSE Loss + λ × OCR Feature Loss
```
Where λ controls the importance of OCR feature similarity.

---

## Dataset Preparation

### Dataset Format

The dataset should be a `.txt` file with semicolon-separated values:

```
<high_res_path>;<low_res_path>;<split>
```

**Example:**
```
/data/HR/plate001.jpg;/data/LR/plate001.jpg;train
/data/HR/plate002.jpg;/data/LR/plate002.jpg;train
/data/HR/plate003.jpg;/data/LR/plate003.jpg;validation
/data/HR/plate004.jpg;/data/LR/plate004.jpg;test
```

### Split Types
- `train`: Training set (recommended: 70%)
- `validation`: Validation set (recommended: 15%)
- `test`: Test set (recommended: 15%)

### Image Requirements

**High-Resolution (HR):**
- Size: 128×64 pixels
- Format: JPG, PNG
- Color: RGB

**Low-Resolution (LR):**
- Size: 32×16 pixels (4x downscale)
- Format: JPG, PNG
- Color: RGB

### Automated Dataset Preparation

Use the provided script for ICPR format datasets (with track_* folders):

```bash
python prepare_dataset.py \
    --dataset-root /path/to/your/raw/dataset \
    --output-dir ./dataset \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**Example: ICPR Brazilian plates**
```bash
python prepare_dataset.py \
    --dataset-root /data/ICPR/train/Scenario-A/Brazilian \
    --output-dir ./dataset
```

**Available arguments:**
- `--dataset-root` (required): Path to raw dataset containing track_* folders
- `--output-dir`: Where to save dataset.txt (default: ./dataset)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

**What the script does:**
1. Scans for all track_* folders in dataset-root
2. Reads annotations.json from each track
3. Pairs HR and LR images (hr-001.png with lr-001.png)
4. Creates metadata .txt files for each HR image
5. Splits into train/val/test sets
6. Generates dataset.txt with all image paths

---

## Training Modes

### Mode 0: Start New Training

Start training from scratch with randomly initialized weights:

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 \
    -m 0
```

**What happens:**
- Creates new checkpoint directory if it doesn't exist
- Initializes network with random weights
- Starts training from epoch 0
- Saves checkpoints to specified directory

### Mode 1: Resume Training

Continue training from a saved checkpoint:

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 \
    -m 1 \
    --model ./checkpoints/backup.pt
```

**What happens:**
- Loads model weights from checkpoint
- Loads optimizer state
- Loads scheduler state
- Loads early stopping counter
- Continues from saved epoch number

### Debug Mode

Test the pipeline on a tiny subset before full training:

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./debug_checkpoints \
    -b 4 \
    -m 0 \
    --debug \
    --debug-samples 20
```

**Features:**
- Uses only 20 training samples (configurable)
- Uses 5 validation samples
- Runs for only 3 epochs
- Disables early stopping
- Perfect for verifying setup

---

## Hyperparameters

### Configurable via Command Line

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Batch Size | `-b, --batch` | Required | Number of images per batch (8 for GPU, 2-4 for CPU) |
| Debug Samples | `--debug-samples` | 20 | Number of samples in debug mode |

### Hardcoded in training.py

You can modify these in the code:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Max Epochs | Line 391 | 200 | Maximum training epochs |
| Learning Rate | Line 367 | 0.0001 | Initial learning rate (Adam) |
| Beta Values | Line 367 | (0.9, 0.999) | Adam optimizer betas |
| Early Stop Patience | Line 371 | 20 | Epochs before early stopping |
| LR Scheduler Patience | Line 368-369 | 10 | Epochs before reducing LR |
| LR Scheduler Factor | Line 368-369 | 0.5 | LR reduction factor |
| Lambda (OCR loss) | Lines 258, 298 | 0.5 | Weight for OCR feature loss |

### Network Architecture Parameters

Defined in `network.py`:

```python
# In Network.__init__() (line 280-285)
AutoEncoder(in_channels=3, out_channels=128, expansion=4)
RDN(num_channels=128, num_features=128, growth_rate=128, num_blocks=32, num_layers=3)
ResidualModule(channels_in=128, out_channels=128)
FeatureModule(channels_in=128, skip_connection_channels=128)
RDN(num_channels=128, num_features=128, growth_rate=128, num_blocks=18, num_layers=3)
```

---

## Monitoring Training

### Console Output

During training, you'll see:

```
Epoch 42 of 200:
TRAINING
100%|████████| 2187/2187 [06:23<00:00, 5.70it/s, Running_loss=0.156, Current_loss=0.152, mse_loss=0.082, features=0.074]

VALIDATION
100%|████████| 469/469 [00:54<00:00, 8.63it/s, Running_loss=0.149, Current_loss=0.148, mse_loss=0.078, features=0.071]

Saving: backup.pt
G validation Loss:  0.14892341
G Training Loss:  0.15634217
```

### Key Metrics

- **Running_loss**: Moving average of total loss
- **Current_loss**: Loss for current batch
- **mse_loss**: Pixel-level MSE loss component
- **features**: OCR feature loss component
- **G validation Loss**: Average validation loss
- **G Training Loss**: Average training loss

### Early Stopping

The training uses early stopping to prevent overfitting:

```
WARNING: Stop counter 5 of 20
```

**What this means:**
- Validation loss hasn't improved for 5 epochs
- Will stop training if no improvement after 20 total epochs
- Best model is always saved

### Learning Rate Scheduling

The learning rate is automatically reduced when validation loss plateaus:

```
Epoch 00052: reducing learning rate of group 0 to 5.0000e-05.
```

This happens after 10 epochs without improvement (configurable).

---

## Weights & Biases Integration

Track training metrics, sample images, and test results with [Weights & Biases](https://wandb.ai):

### Setup

```bash
# Install wandb (optional)
pip install wandb
wandb login  # First time only
```

### Training with W&B

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 -m 0 \
    --wandb-project my-lpr-project \
    --wandb-run-name experiment-001 \
    --wandb-log-interval 5
```

### Testing with W&B

Testing automatically resumes the training run if a run ID was saved:

```bash
python testing.py \
    -t ../dataset/dataset.txt \
    -s ./results \
    -b 4 \
    --model ./checkpoints/backup.pt
```

### Disable W&B

```bash
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 8 -m 0 --disable-wandb
```

### Tracked Metrics

| Phase | Metrics |
|-------|---------|
| Training | Batch losses (total, MSE, features), running averages |
| Validation | Batch losses and running averages per epoch |
| Sample Images | LR/HR/SR comparison every N epochs |
| Testing | PSNR, SSIM, OCR accuracy distributions, detailed results table |

### W&B Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--wandb-project` | Project name | `lpr-super-resolution` |
| `--wandb-run-name` | Custom run name | Auto-generated |
| `--wandb-entity` | Team/organization name | None |
| `--wandb-log-interval` | Image logging frequency (epochs) | 5 |
| `--disable-wandb` | Disable W&B logging | False |
| `--wandb-run-id` | Resume specific run (testing only) | None |

**Note:** W&B is completely optional. Training and testing work normally without it installed or when disabled with `--disable-wandb`.

---

## Checkpointing

### Checkpoint Contents

Saved checkpoints (`backup.pt`) contain:

```python
{
    'epoch': 42,                    # Current epoch number
    'model_state_dict': ...,        # Network weights
    'optimizer_state_dict': ...,    # Optimizer state
    'scheduler_state_dict': ...,    # LR scheduler state
    'val_loss': 0.149,              # Best validation loss
    'train_loss': 0.156,            # Latest training loss
    'early_stopping_counter': 5,    # Early stopping state
}
```

### Checkpoint Location

- Default: `./checkpoints/backup.pt`
- Specified with `-s` flag
- File size: ~1.9 GB (network has ~500M parameters)

### Checkpoint Frequency

- Saved after every epoch
- Only the best model (lowest validation loss) is kept
- Automatically overwrites previous checkpoint

### Loading Checkpoints

To resume training:

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 \
    -m 1 \
    --model ./checkpoints/backup.pt
```

To test with a checkpoint:

```bash
python testing.py \
    -t ../dataset/dataset.txt \
    -s ./results \
    -b 4 \
    --model ./checkpoints/backup.pt
```

---

## Performance Tips

### GPU Training

**Recommended Settings:**
```bash
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 8 -m 0
```

- Batch size: 8-16 (depending on GPU memory)
- Expected speed: ~5-10 it/s (iterations per second)
- Time per epoch: ~5-10 minutes (25k images)

**GPU Memory Usage:**
- Model: ~2 GB
- Batch size 8: ~6-8 GB total
- Batch size 16: ~10-12 GB total

### CPU Training

**Recommended Settings:**
```bash
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 2 -m 0
```

- Batch size: 2-4 (CPU has no memory constraints)
- Expected speed: ~0.1-0.5 it/s
- Time per epoch: ~2-6 hours (25k images)

**CPU Considerations:**
- Much slower than GPU (~10-50x)
- Suitable for debugging and small datasets
- Can use larger batch sizes (not limited by VRAM)

### Optimizing Training Speed

1. **Use GPU if available**
   ```bash
   # Verify GPU is detected
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Increase batch size** (if memory allows)
   - Larger batches = better GPU utilization
   - But: requires more memory

3. **Reduce dataset size for testing**
   - Use `--debug` mode during development
   - Only use full dataset for final training

4. **Use mixed precision** (advanced)
   - Not currently implemented
   - Could reduce memory and increase speed

---

## Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. Reduce batch size: `-b 4` or `-b 2`
2. Close other GPU applications
3. Use smaller network (modify network.py)
4. Enable gradient checkpointing (requires code changes)

### CUDA Not Available

```
⚠️  CUDA not available, training on CPU (will be slower)
```

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA installation:
   ```python
   import torch
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   ```
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```
4. If GPU unavailable, training will work on CPU (slower)

### OCR Model Loading Errors

```
FileNotFoundError: ../saved_models/RodoSol-SR/model.json
TypeError: Could not locate class 'Model'
```

**Solutions:**
1. Verify OCR models are in correct location:
   ```bash
   ls ../saved_models/RodoSol-SR/
   # Should show: model.json, weights.hdf5
   ```
2. Check TensorFlow/Keras versions:
   ```bash
   pip install tensorflow==2.15.1 keras==2.15.0
   ```
3. Run from `Proposed/` directory (OCR paths are relative)

### Training Not Converging

**Symptoms:**
- Loss stays high (>0.5) after many epochs
- Validation loss higher than training loss
- Generated images look blurry

**Solutions:**
1. Check dataset quality (HR/LR correspondence)
2. Verify image dimensions (HR: 128×64, LR: 32×16)
3. Increase training epochs
4. Adjust learning rate (modify training.py line 367)
5. Check OCR model is loading correctly

### Validation Loss Increasing

**Symptoms:**
- Training loss decreasing
- Validation loss increasing
- Model is overfitting

**Solutions:**
1. Early stopping will handle this automatically
2. Use more training data
3. Add data augmentation (see __dataset__.py)
4. Reduce model complexity

### Slow Training on GPU

**Check GPU utilization:**
```bash
nvidia-smi
# GPU-Util should be >80%
```

**If GPU underutilized:**
1. Increase batch size
2. Reduce num_workers in dataloader (training.py line 358)
3. Check data loading isn't bottleneck
4. Verify images are on SSD, not HDD

---

## Advanced Training Options

### Custom Loss Weights

Modify the lambda value in training.py:

```python
# Line 258 and 298
loss_total = loss_mse + 0.5 * loss_features  # Change 0.5 to adjust weight
```

- Higher lambda (e.g., 1.0): More emphasis on OCR features
- Lower lambda (e.g., 0.1): More emphasis on pixel accuracy

### Data Augmentation

Augmentation is configured in `__dataset__.py` (lines 89-102):

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Add more augmentations here
])
```

Available augmentations:
- Rotation
- Scaling
- Gaussian noise
- Blur
- Elastic transforms

### Using Different OCR Models

To use PKU-SR instead of RodoSol-SR:

```python
# In training.py, line 360
ocr_path = Path('../saved_models/PKU-SR')  # Change from RodoSol-SR
```

---

## Best Practices

1. **Always test with debug mode first**
   ```bash
   python training.py ... --debug --debug-samples 20
   ```

2. **Monitor validation loss closely**
   - Should decrease along with training loss
   - If diverging, model is overfitting

3. **Save multiple checkpoints** (optional)
   - Modify code to save top-k models
   - Useful for ensemble or model selection

4. **Use tensorboard for visualization** (optional)
   - Add tensorboard logging (requires code changes)
   - Visualize loss curves and sample images

5. **Validate on real-world data**
   - Test set should represent actual use cases
   - Check PSNR, SSIM, and OCR accuracy metrics

---

## Expected Training Results

After ~50-100 epochs on 25k images:

- **Training Loss:** 0.10-0.15
- **Validation Loss:** 0.12-0.18
- **PSNR:** 25-30 dB
- **OCR Accuracy Improvement:** 10-20% over bicubic interpolation

**Note:** Results vary based on dataset quality and characteristics.

---

## Next Steps

After training completes:

1. **Evaluate on test set:**
   ```bash
   python testing.py -t ../dataset/dataset.txt -s ./results -b 4 --model ./checkpoints/backup.pt
   ```

2. **Compare with baseline** (bicubic interpolation)

3. **Fine-tune on specific plate types** if needed

4. **Deploy model** for inference on new images

For Google Colab training with free GPU, see `COLAB_GUIDE.md`.
