#!/bin/bash
set -e

echo "=============================================="
echo "LPR Super-Resolution Training"
echo "=============================================="

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------------------------
# OCR Models Setup
# -----------------------------------------------------------------------------
OCR_MODELS_DIR="$REPO_DIR/saved_models"
OCR_MODELS_URL="https://github.com/Valfride/lpr-rsr-ext/releases/download/OCR_pre-trained_models/saved_models.zip"

if [ -d "$OCR_MODELS_DIR/RodoSol-SR" ] && [ -d "$OCR_MODELS_DIR/PKU-SR" ]; then
    echo "✅ OCR models already exist at $OCR_MODELS_DIR"
else
    echo "📥 OCR models not found. Downloading..."
    wget -q "$OCR_MODELS_URL" -O /tmp/saved_models.zip
    unzip -q /tmp/saved_models.zip -d "$REPO_DIR"
    rm -f /tmp/saved_models.zip
    echo "✅ OCR models downloaded to $OCR_MODELS_DIR"
fi

# -----------------------------------------------------------------------------
# Dataset Setup
# -----------------------------------------------------------------------------
DATASET_DIR="/root/experiments/ICPR_dataset"
DATASET_ZIP="/root/experiments/ICPR_dataset.zip"
GDRIVE_FILE_ID="1RMfWdDboyVYtszoiwda7rFmmf2OTX928"

# Check if dataset exists
if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    echo "✅ Dataset already exists at $DATASET_DIR"
else
    echo "📥 Dataset not found. Downloading from Google Drive..."

    # Download from Google Drive
    gdown --id "$GDRIVE_FILE_ID" -O "$DATASET_ZIP"

    echo "📦 Extracting dataset..."
    mkdir -p "$DATASET_DIR"
    unzip -q "$DATASET_ZIP" -d "$DATASET_DIR"

    # Clean up zip file to save space
    rm -f "$DATASET_ZIP"

    echo "✅ Dataset downloaded and extracted to $DATASET_DIR"
fi

# List dataset structure for verification
echo "📂 Dataset structure:"
ls -la "$DATASET_DIR" | head -10

# -----------------------------------------------------------------------------
# Update dataset.txt paths
# -----------------------------------------------------------------------------
echo "🔄 Updating dataset paths..."

# Create updated dataset.txt with correct paths
# Original paths: /home/ccds/moshi/ICPR/dataset/...
# New paths: /root/experiments/ICPR_dataset/...
ORIGINAL_DATASET_TXT="$REPO_DIR/dataset/dataset.txt"
UPDATED_DATASET_TXT="$REPO_DIR/dataset/dataset_modal.txt"

if [ -f "$ORIGINAL_DATASET_TXT" ]; then
    # Original: /home/ccds/moshi/ICPR/dataset/train/Scenario-A/Brazilian/...
    # Target:   /root/experiments/ICPR_dataset/Brazilian/...
    sed 's|/home/ccds/moshi/ICPR/dataset/train/Scenario-A|/root/experiments/ICPR_dataset|g' \
        "$ORIGINAL_DATASET_TXT" > "$UPDATED_DATASET_TXT"
    echo "✅ Created updated dataset file: $UPDATED_DATASET_TXT"
    echo "📋 Sample path: $(head -1 $UPDATED_DATASET_TXT | cut -d';' -f1)"
else
    echo "⚠️  Original dataset.txt not found, using as-is"
    UPDATED_DATASET_TXT="$ORIGINAL_DATASET_TXT"
fi

# -----------------------------------------------------------------------------
# W&B Setup
# -----------------------------------------------------------------------------
echo "🔑 Logging into Weights & Biases..."
wandb login

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
cd "$REPO_DIR/Proposed"

# Create checkpoints directory
mkdir -p checkpoints

echo "🚀 Starting training..."

# Run training with W&B enabled
# Mode 0: Train from scratch
# Mode 1: Resume from checkpoint
python training.py \
    -t "$UPDATED_DATASET_TXT" \
    -s ./checkpoints \
    -b 16 \
    -m 0 \
    --wandb-project lpr-super-resolution

echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
