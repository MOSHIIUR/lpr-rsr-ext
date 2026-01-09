#!/bin/bash
set -e

echo "=============================================="
echo "LPR Super-Resolution Training"
echo "=============================================="

# Navigate to Proposed directory
cd Proposed

# Create checkpoints directory
mkdir -p checkpoints

# Run training
# Mode 0: Train from scratch
# Mode 1: Resume from checkpoint
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 16 \
    -m 0

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
