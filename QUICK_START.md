# Quick Start Guide for Custom Dataset

## Step 1: Prepare Your Dataset

Run the dataset preparation script:

```bash
cd /home/ccds/moshi/lpr-rsr-ext
python prepare_dataset.py \
    --dataset-root /home/ccds/moshi/ICPR/dataset/train/Scenario-A/Brazilian \
    --output-dir ./dataset
```

This will:
- ✅ Process all `track_*` folders
- ✅ Read `annotations.json` files
- ✅ Create 70/15/15 train/val/test split
- ✅ Generate `dataset.txt` file
- ✅ Create metadata `.txt` files for each HR image

**Output location:** `./dataset/dataset.txt`

## Step 2: Configure Image Dimensions

Edit `Proposed/__dataset__.py` and set:
```python
IMG_LR = (40, 20)  # Adjust based on your LR image size
IMG_HR = (160, 80)  # Adjust based on your HR image size
```

And in the `customDataset` class:
```python
self.aspect_ratio = 2.0  # Adjust based on your images (width/height)
```

## Step 3: Update OCR Model Path (if needed)

Edit `Proposed/training.py` (line 339) and `Proposed/testing.py` (line 22):
```python
path_ocr = Path('./saved_models/RodoSol-SR')  # or your OCR model path
```

## Step 4: Train the Model

### Basic Training (without W&B)
```bash
cd Proposed
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 16 \
    -m 0
```

### With Weights & Biases (Optional)
Track experiments with W&B for better visualization and experiment management:

```bash
# Install wandb (first time only)
pip install wandb
wandb login  # Enter your API key from https://wandb.ai/authorize

# Train with W&B tracking
cd Proposed
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 16 \
    -m 0 \
    --wandb-project my-lpr-project \
    --wandb-run-name experiment-001
```

**W&B Features:**
- Real-time loss curves and metrics visualization
- Automatic run resumption for testing phase
- Compare multiple experiments easily
- Sample images logged every N epochs

## Step 5: Test the Model

```bash
cd Proposed
python testing.py \
    -t ../dataset/dataset.txt \
    -s ./results \
    -b 16 \
    --model ./checkpoints/bestmodel.pt
```

**Note:** If you used W&B for training, testing will automatically resume the same run and log test metrics.

## Troubleshooting

1. **If you get "file not found" errors**: Make sure all paths are absolute or relative to the current directory
2. **If metadata files are missing**: The script creates them automatically, but check that JSON files exist in each track folder
3. **If image dimensions don't match**: Update `IMG_LR` and `IMG_HR` in `__dataset__.py`
4. **If OCR model not found**: Download from the link in README.md and extract to `./saved_models/`



