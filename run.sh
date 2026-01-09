#!/bin/bash
set -e

echo "=============================================="
echo "LPR Super-Resolution Training"
echo "=============================================="

# W&B login (uses WANDB_API_KEY from environment)
wandb login

# Navigate to Proposed directory
cd Proposed

# Create checkpoints directory
mkdir -p checkpoints

# Run training with W&B enabled
# Mode 0: Train from scratch
# Mode 1: Resume from checkpoint
python training.py \
    -t ../dataset/dataset.txt \
    -s ./checkpoints \
    -b 16 \
    -m 0 \
    --wandb-project lpr-super-resolution

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
