# Training and Testing Guide

## Prerequisites

1. **Dataset**: You need a dataset file (`.txt`) with the following format:
   ```
   /path/to/HR_image1.jpg;/path/to/LR_image1.jpg;training
   /path/to/HR_image2.jpg;/path/to/LR_image2.jpg;validation
   /path/to/HR_image3.jpg;/path/to/LR_image3.jpg;testing
   ```
   - Each line contains: `HR_image_path;LR_image_path;split_type`
   - Split types: `training`, `validation`, or `testing`
   - Each HR image must have a corresponding `.txt` file with metadata (plate number, layout, type)

### Preparing Custom ICPR Dataset

If you have a custom dataset with the following structure:
```
Brazilian/
  track_00001/
    hr-001.png, hr-002.png, ..., hr-005.png
    lr-001.png, lr-002.png, ..., lr-005.png
    annotations.json
  track_00002/
    ...
```

Use the provided script to prepare it:

```bash
python prepare_dataset.py \
    --dataset-root /home/ccds/moshi/ICPR/dataset/train/Scenario-A/Brazilian \
    --output-dir ./dataset \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

This script will:
- Read all `track_*` folders and their `annotations.json` files
- Create train/validation/test split (default: 70/15/15)
- Generate `dataset.txt` file in the required format
- Create `.txt` metadata files for each HR image automatically
- Save everything to the output directory

**Output:**
- `dataset.txt`: Main dataset file for training/testing
- `training.txt`, `validation.txt`, `testing.txt`: Separate split files
- Metadata `.txt` files created alongside HR images

2. **OCR Model**: Download pre-trained OCR models from [here](https://github.com/Valfride/lpr-rsr-ext/releases/download/OCR_pre-trained_models/saved_models.zip) and extract to `./saved_models/`

3. **Configuration**: Update image dimensions in `__dataset__.py`:
   ```python
   # For RodoSol-SR
   IMG_LR = (40, 20)
   IMG_HR = (160, 80)
   # aspect_ratio = 2.0
   
   # For PKU-SR
   # IMG_LR = (48, 16)
   # IMG_HR = (192, 64)
   # aspect_ratio = 3.0
   ```

## Training

### Step 1: Navigate to the Proposed directory
```bash
cd Proposed
```

### Step 2: Train from scratch (Mode 0)
```bash
python training.py \
    -t /path/to/your/dataset.txt \
    -s /path/to/save/models \
    -b 16 \
    -m 0
```

**Parameters:**
- `-t, --samples`: Path to your dataset `.txt` file
- `-s, --save`: Directory where models will be saved
- `-b, --batch`: Batch size (e.g., 16, 32)
- `-m, --mode`: Training mode
  - `0` = Train from scratch (creates new model)
  - `1` = Resume training from checkpoint (requires `--model`)

### Step 3: Resume training (Mode 1)
If you need to resume from a checkpoint:
```bash
python training.py \
    -t /path/to/your/dataset.txt \
    -s /path/to/save/models \
    -b 16 \
    -m 1 \
    --model /path/to/save/models/bestmodel.pt
```

**What happens during training:**
- Models are saved as `bestmodel.pt` (best validation loss) and `backup.pt` (latest checkpoint)
- Training runs for up to 200 epochs with early stopping (patience=20)
- Learning rate is reduced on plateau (factor=0.8, patience=2)
- Loss combines MSE and OCR feature extraction (L1 loss on OCR features)

## Testing

### Step 1: Navigate to the Proposed directory
```bash
cd Proposed
```

### Step 2: Run testing
```bash
python testing.py \
    -t /path/to/your/dataset.txt \
    -s /path/to/save/results \
    -b 16 \
    --model /path/to/save/models/bestmodel.pt
```

**Parameters:**
- `-t, --samples`: Path to your dataset `.txt` file (must contain `testing` split)
- `-s, --save`: Directory where results will be saved
- `-b, --batch`: Batch size for testing
- `--model`: Path to the trained model (`.pt` file)

**Output files:**
- `eval.csv`: Comprehensive evaluation metrics
- `resultsHR.csv`: Accuracy distribution for HR images
- `resultsLR.csv`: Accuracy distribution for LR images
- `resultsSR.csv`: Accuracy distribution for SR images
- `HR_*.jpg/png`: High-resolution ground truth images
- `LR_*.jpg/png`: Low-resolution input images
- `SR_*.jpg/png`: Super-resolved output images

**Metrics computed:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- OCR Accuracy (character-level Levenshtein distance)

## Example Workflow

### For Custom ICPR Dataset:

```bash
# 1. Prepare the dataset (creates dataset.txt and metadata files)
python prepare_dataset.py \
    --dataset-root /home/ccds/moshi/ICPR/dataset/train/Scenario-A/Brazilian \
    --output-dir ./dataset

# 2. Train the model
cd Proposed
python training.py -t ../dataset/dataset.txt -s ./checkpoints -b 16 -m 0

# 3. Wait for training to complete (or stop early)
# Best model will be saved as: ./checkpoints/bestmodel.pt

# 4. Test the model
python testing.py -t ../dataset/dataset.txt -s ./results -b 16 --model ./checkpoints/bestmodel.pt

# 5. Check results
ls ./results/
cat ./results/eval.csv
```

### For Pre-formatted Dataset:

```bash
# 1. Prepare your dataset file (dataset.txt)
# Format: HR_path;LR_path;split_type

# 2. Train the model
cd Proposed
python training.py -t ../dataset.txt -s ./checkpoints -b 16 -m 0

# 3. Wait for training to complete (or stop early)
# Best model will be saved as: ./checkpoints/bestmodel.pt

# 4. Test the model
python testing.py -t ../dataset.txt -s ./results -b 16 --model ./checkpoints/bestmodel.pt

# 5. Check results
ls ./results/
cat ./results/eval.csv
```

## Important Notes

1. **OCR Model Path**: Update the OCR model path in `training.py` (line 339) and `testing.py` (line 22) if needed:
   ```python
   path_ocr = Path('./saved_models/your-ocr-model-name')
   ```

2. **GPU**: The code uses CUDA by default. Make sure you have a GPU available or modify the code to use CPU.

3. **Image Dimensions**: Ensure `IMG_LR` and `IMG_HR` in `__dataset__.py` match your dataset's aspect ratio.

4. **Dataset Format**: Each HR image must have a corresponding `.txt` file with metadata:
   ```
   Type: car
   Plate: ABC1234
   Layout: ABC-1234
   Points: x1,y1 x2,y2 x3,y3 x4,y4
   ```

5. **Early Stopping**: Training stops automatically if validation loss doesn't improve for 20 epochs.

6. **Checkpoints**: Regular backups are saved as `backup.pt`. The best model is saved as `bestmodel.pt`.

