# Quick Start Guide - LPR Super-Resolution

Get up and running with the License Plate Recognition Super-Resolution model in minutes.

## Prerequisites

- Conda or Anaconda installed
- NVIDIA GPU (optional, but recommended for faster training)
- 25,000+ license plate images prepared in ICPR format

## 1. Environment Setup (5 minutes)

```bash
# Clone or navigate to the repository
cd lpr-rsr-ext/Proposed

# Create conda environment
conda create -n lpr-rsr python=3.9 -y
conda activate lpr-rsr

# Install PyTorch with CUDA support (for GPU)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install tensorflow==2.15.1
pip install tf-keras  # Keras 2 compatibility for OCR models
pip install opencv-python albumentations pandas tqdm scikit-learn
```

## 2. Prepare Dataset (2 minutes)

Your dataset should be a text file with semicolon-separated paths:

```
/path/to/high_res_image1.jpg;/path/to/low_res_image1.jpg;train
/path/to/high_res_image2.jpg;/path/to/low_res_image2.jpg;train
/path/to/high_res_image3.jpg;/path/to/low_res_image3.jpg;validation
/path/to/high_res_image4.jpg;/path/to/low_res_image4.jpg;test
```

If you have ICPR format dataset (with track_* folders), use the provided script:

```bash
cd ..
python prepare_dataset.py \
    --dataset-root /path/to/your/raw/dataset \
    --output-dir ./dataset

# Example: For ICPR Brazilian plates
python prepare_dataset.py \
    --dataset-root /path/to/ICPR/train/Scenario-A/Brazilian \
    --output-dir ./dataset

# Creates: dataset/dataset.txt with train/val/test split (70/15/15 by default)
```

## 3. Download OCR Models (1 minute)

Extract pre-trained OCR models to `saved_models/`:

```
saved_models/
├── RodoSol-SR/
│   ├── model.json
│   └── weights.hdf5
└── PKU-SR/
    ├── model.json
    └── weights.hdf5
```

## 4. Test Setup with Debug Mode (2-5 minutes)

Verify everything works before full training:

```bash
cd Proposed
python training.py \
    -t ../dataset/dataset.txt \
    -s ./debug_checkpoints \
    -b 4 \
    -m 0 \
    --debug \
    --debug-samples 20
```

**Expected Output:**
```
🖥️  Using device: cuda:0 (or cpu)
🔍 DEBUG MODE ENABLED
   Training samples: 20
   Max epochs: 3
   Early stopping: DISABLED

Epoch 0 of 3:
  TRAINING: 100% [loss values shown]
  VALIDATION: 100% [loss values shown]
  ✓ Checkpoint saved

... (epochs 1-2) ...

Total execution time: 00:05:00 (varies)
```

## 5. Start Full Training

Once debug mode succeeds, start full training:

```bash
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 \
    -m 0
```

**Training will:**
- Run for up to 200 epochs
- Use early stopping (patience=20)
- Save best model to `checkpoints/backup.pt`
- Display progress bars with loss metrics

### Optional: Enable W&B Tracking

Track experiments with [Weights & Biases](https://wandb.ai):

```bash
pip install wandb
wandb login

python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 8 -m 0 \
    --wandb-project my-lpr-project
```

See `TRAINING_GUIDE.md` for full W&B options.

## 6. Test Your Model

After training, test on validation/test images:

```bash
python testing.py \
    -t ../dataset/dataset.txt \
    -s ./results \
    -b 4 \
    --model ./checkpoints/backup.pt
```

Results are saved to `./results/` directory.

---

## Common Issues

### CUDA not available
```
⚠️  CUDA not available, training on CPU (will be slower)
```
**Solution:** Training will work but be slower. To use GPU, ensure CUDA drivers are installed and PyTorch is compiled with CUDA support.

### OCR model not found
```
FileNotFoundError: ../saved_models/RodoSol-SR/model.json
```
**Solution:** Extract OCR models to the correct path relative to `Proposed/` directory.

### Out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size: `-b 4` or `-b 2`

---

## Command Reference

### Training Arguments
- `-t, --samples`: Path to dataset.txt file (required)
- `-s, --save`: Directory for saving checkpoints (required)
- `-b, --batch`: Batch size (required, try 8 for GPU, 2-4 for CPU)
- `-m, --mode`: 0=new training, 1=resume training (required)
- `--model`: Path to checkpoint for resuming (required if mode=1)
- `--debug`: Enable debug mode with tiny dataset
- `--debug-samples`: Number of samples in debug mode (default: 20)
- `--wandb-project`: W&B project name (optional)
- `--disable-wandb`: Disable W&B logging

### Testing Arguments
- `-t, --samples`: Path to dataset.txt file (required)
- `-s, --save`: Directory for saving results (required)
- `-b, --batch`: Batch size (required)
- `--model`: Path to trained model checkpoint (required)

---

## What's Next?

- See `TRAINING_GUIDE.md` for detailed training options and tips
- See `COLAB_GUIDE.md` for running on Google Colab with free GPU
- Check the main `README.md` for architecture details

---

**Quick Tip:** Always run `--debug` mode first to verify your setup before starting long training runs!
